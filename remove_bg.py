import argparse
import os
from contextlib import nullcontext
from glob import glob
import time

import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from config import Config
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import check_state_dict


def parse_resolution(resolution, default_size):
    if resolution in [None, "", "None"]:
        return None
    if resolution == "config.size":
        return default_size
    try:
        w, h = resolution.lower().split("x")
        return int(w), int(h)
    except Exception as e:
        raise ValueError(f"Invalid --resolution '{resolution}'. Use WxH, e.g. 1024x1024.") from e


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


def sync_device(device):
    device_str = str(device)
    if "cuda" in device_str and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_str == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def run_inference(model, tensor, autocast_ctx):
    with autocast_ctx, torch.no_grad():
        return model(tensor)[-1].sigmoid().to(torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Remove background using BiRefNet.")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to .pth checkpoint")
    parser.add_argument("--input", required=True, type=str, help="Input image path or folder")
    parser.add_argument("--output", required=True, type=str, help="Output image path or folder")
    parser.add_argument("--output_mask", default="", type=str, help="Optional output mask path or folder")
    parser.add_argument("--resolution", default="", type=str, help="Resize to WxH before inference, or 'config.size'")
    parser.add_argument("--device", default="", type=str, help="cuda:0, cuda, or cpu")
    parser.add_argument("--benchmark", action="store_true", help="Print timing for preprocessing/inference/post")
    parser.add_argument("--repeat", default=1, type=int, help="Repeat inference N times for average timing")
    parser.add_argument("--mask_mode", default="soft", choices=["soft", "hard"], help="Soft alpha or hard threshold mask")
    parser.add_argument("--mask_threshold", default=0.5, type=float, help="Threshold for hard mask")
    parser.add_argument("--alpha_power", default=1.0, type=float, help="Sharpen alpha by power (>1 sharper edges)")
    parser.add_argument("--edge_refine", action="store_true", help="Refine mask edges with simple morphology")
    parser.add_argument("--edge_refine_size", default=3, type=int, help="Kernel size for edge refine (odd int)")
    parser.add_argument("--max_side", default=0, type=int, help="Optional max size for longer image side (keep aspect)")
    args = parser.parse_args()

    config = Config()
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = f"cuda:{config.device}" if isinstance(config.device, int) else "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if config.precisionHigh:
        torch.set_float32_matmul_precision("high")

    resolution = parse_resolution(args.resolution, config.size)

    mixed_precision = config.mixed_precision
    if mixed_precision == "fp16":
        mixed_dtype = torch.float16
    elif mixed_precision == "bf16":
        mixed_dtype = torch.bfloat16
    else:
        mixed_dtype = None
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=mixed_dtype) if mixed_dtype and "cuda" in str(device) else nullcontext()

    model = load_model(config, args.ckpt, device)

    def process_one(input_path, output_path, output_mask_path):
        t0 = time.perf_counter()
        image = Image.open(input_path).convert("RGB")
        orig_size = image.size
        if resolution:
            image = image.resize(resolution, Image.BILINEAR)
        elif args.max_side and max(orig_size) > args.max_side:
            scale = args.max_side / float(max(orig_size))
            new_w = int(orig_size[0] * scale)
            new_h = int(orig_size[1] * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)

        image, crop_box = pad_to_multiple_of_32(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0).to(device)
        t_pre = time.perf_counter()

        repeat = max(1, args.repeat)
        sync_device(device)
        t_inf_start = time.perf_counter()
        pred = None
        for _ in range(repeat):
            pred = run_inference(model, tensor, autocast_ctx)
        sync_device(device)
        t_inf_end = time.perf_counter()

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

        if args.alpha_power != 1.0:
            pred = pred.clamp(0, 1).pow(args.alpha_power)
        if args.mask_mode == "hard":
            pred = (pred >= args.mask_threshold).float()
        mask = (pred.clamp(0, 1) * 255).byte().numpy()
        mask_img = Image.fromarray(mask, mode="L")
        if args.edge_refine:
            k = max(3, args.edge_refine_size | 1)
            mask_img = mask_img.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
            mask_img = mask_img.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))

        rgba = Image.open(input_path).convert("RGBA")
        rgba.putalpha(mask_img)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out_ext = os.path.splitext(output_path)[1].lower()
        if out_ext in [".jpg", ".jpeg"]:
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(white_bg, rgba).convert("RGB")
            rgb.save(output_path)
        else:
            rgba.save(output_path)

        if output_mask_path:
            os.makedirs(os.path.dirname(output_mask_path) or ".", exist_ok=True)
            mask_img.save(output_mask_path)
        t_post = time.perf_counter()

        if args.benchmark:
            pre_ms = (t_pre - t0) * 1000
            inf_ms = (t_inf_end - t_inf_start) * 1000
            post_ms = (t_post - t_inf_end) * 1000
            total_ms = (t_post - t0) * 1000
            avg_inf_ms = inf_ms / repeat
            fps = 1000.0 / avg_inf_ms if avg_inf_ms > 0 else 0.0
            print(f"[Timing] {os.path.basename(input_path)}")
            print(f"[Timing] preprocess: {pre_ms:.1f} ms")
            print(f"[Timing] inference:  {inf_ms:.1f} ms (avg {avg_inf_ms:.1f} ms, {fps:.2f} FPS)")
            print(f"[Timing] post:       {post_ms:.1f} ms")
            print(f"[Timing] total:      {total_ms:.1f} ms")

    if os.path.isdir(args.input):
        input_dir = args.input
        output_dir = args.output
        mask_dir = args.output_mask if args.output_mask else os.path.join(output_dir, "masks")
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        paths = []
        for ext in exts:
            paths.extend(glob(os.path.join(input_dir, f"*{ext}")))
            paths.extend(glob(os.path.join(input_dir, f"*{ext.upper()}")))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No images found in {input_dir}")
        print(f"Found {len(paths)} image(s) in {input_dir}")
        for idx, p in enumerate(paths, start=1):
            stem = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(output_dir, f"{stem}.png")
            mask_path = os.path.join(mask_dir, f"{stem}-mask.png") if mask_dir else ""
            print(f"[{idx}/{len(paths)}] {os.path.basename(p)}")
            process_one(p, out_path, mask_path)
        return

    if os.path.isdir(args.output):
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(args.output, f"{stem}.png")
        if args.output_mask:
            output_mask_path = os.path.join(args.output_mask, f"{stem}-mask.png") if os.path.isdir(args.output_mask) else args.output_mask
        else:
            output_mask_path = ""
    else:
        output_path = args.output
        output_mask_path = args.output_mask

    process_one(args.input, output_path, output_mask_path)


if __name__ == "__main__":
    main()
