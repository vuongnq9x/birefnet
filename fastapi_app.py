import argparse
import os
from contextlib import nullcontext
from io import BytesIO
import time

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from config import Config
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import check_state_dict


def pad_to_multiple_of_32(pil_img):
    w, h = pil_img.size
    pad_w = (32 - w % 32) % 32
    pad_h = (32 - h % 32) % 32
    if pad_w == 0 and pad_h == 0:
        return pil_img, (0, 0, w, h)
    new_w = w + pad_w
    new_h = h + pad_h
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    canvas.paste(pil_img, (0, 0))
    return canvas, (0, 0, w, h)


def select_device(config, forced=""):
    if forced:
        return forced
    if torch.cuda.is_available():
        return f"cuda:{config.device}" if isinstance(config.device, int) else "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(config, ckpt_path, device):
    if config.model == "BiRefNet":
        model = BiRefNet(bb_pretrained=False)
    elif config.model == "BiRefNetC2F":
        model = BiRefNetC2F(bb_pretrained=False)
    else:
        raise ValueError(f"Unsupported config.model: {config.model}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_app(ckpt_path, device_override=""):
    config = Config()
    device = select_device(config, device_override)

    if config.precisionHigh:
        torch.set_float32_matmul_precision("high")

    mixed_precision = config.mixed_precision
    if mixed_precision == "fp16":
        mixed_dtype = torch.float16
    elif mixed_precision == "bf16":
        mixed_dtype = torch.bfloat16
    else:
        mixed_dtype = None

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=mixed_dtype)
        if mixed_dtype and "cuda" in str(device)
        else nullcontext()
    )

    model = load_model(config, ckpt_path, device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    app = FastAPI(title="BiRefNet Mask API")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(device)}

    @app.post("/mask")
    async def mask(
        file: UploadFile = File(...),
        max_side: int = Query(0, ge=0),
        mask_mode: str = Query("soft"),
        mask_threshold: float = Query(0.5, ge=0.0, le=1.0),
        alpha_power: float = Query(1.0, ge=0.1, le=5.0),
        edge_refine: bool = Query(False),
        edge_refine_size: int = Query(3, ge=3),
        mask_rgba: bool = Query(False),
        timing: bool = Query(False),
    ):
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        t0 = time.perf_counter()
        try:
            image = Image.open(BytesIO(data))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        image = ImageOps.exif_transpose(image).convert("RGB")
        orig_size = image.size

        if max_side and max(orig_size) > max_side:
            scale = max_side / float(max(orig_size))
            new_w = int(orig_size[0] * scale)
            new_h = int(orig_size[1] * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)

        image, crop_box = pad_to_multiple_of_32(image)
        tensor = transform(image).unsqueeze(0).to(device)

        with autocast_ctx, torch.no_grad():
            pred = model(tensor)[-1].sigmoid().to(torch.float32)

        pred = torch.nn.functional.interpolate(
            pred,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=True
        ).squeeze(0).squeeze(0).cpu()
        pred = pred[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        if pred.shape[1] != orig_size[0] or pred.shape[0] != orig_size[1]:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=orig_size[::-1],
                mode="bilinear",
                align_corners=True
            ).squeeze(0).squeeze(0)

        if mask_mode not in ("soft", "hard"):
            raise HTTPException(status_code=400, detail="mask_mode must be 'soft' or 'hard'")

        if alpha_power != 1.0:
            pred = pred.clamp(0, 1).pow(alpha_power)
        if mask_mode == "hard":
            pred = (pred >= mask_threshold).float()

        mask = (pred.clamp(0, 1) * 255).byte().numpy()
        mask_img = Image.fromarray(mask, mode="L")
        if edge_refine:
            k = max(3, edge_refine_size | 1)
            mask_img = mask_img.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
            mask_img = mask_img.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))

        if mask_rgba:
            mask_rgba_img = Image.new("RGBA", mask_img.size, (255, 255, 255, 0))
            mask_rgba_img.putalpha(mask_img)
            out_img = mask_rgba_img
        else:
            out_img = mask_img

        buf = BytesIO()
        out_img.save(buf, format="PNG")
        buf.seek(0)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        headers = {
            "Content-Disposition": f'attachment; filename="{os.path.splitext(file.filename or "mask")[0]}_mask.png"'
        }
        if timing:
            return JSONResponse({
                "elapsed_ms": elapsed_ms,
                "orig_size": {"w": orig_size[0], "h": orig_size[1]},
                "device": str(device),
            })

        headers["X-Elapsed-MS"] = f"{elapsed_ms:.3f}"
        headers["X-Device"] = str(device)
        headers["X-Orig-Size"] = f"{orig_size[0]}x{orig_size[1]}"
        return StreamingResponse(buf, media_type="image/png", headers=headers)

    return app


def create_app():
    ckpt = os.getenv("BIREfNET_CKPT") or os.getenv("CKPT") or ""
    if not ckpt:
        raise RuntimeError("Missing checkpoint. Set BIREfNET_CKPT or CKPT env var.")
    device = os.getenv("BIREfNET_DEVICE") or ""
    return build_app(ckpt, device_override=device)


# For uvicorn "fastapi_app:app"
try:
    app = create_app()
except Exception:
    app = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run BiRefNet FastAPI server")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to .pth checkpoint")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", default="", type=str, help="cuda:0, cuda, mps, or cpu")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    app = build_app(args.ckpt, device_override=args.device)
    uvicorn.run(app, host=args.host, port=args.port)
