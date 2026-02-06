import os
import random

from locust import HttpUser, task, between
try:
    from locust.exception import StopUser
except Exception:  # fallback for older locust
    StopUser = Exception


class MaskUser(HttpUser):
    wait_time = between(0, 0)

    def _load_paths(self):
        img_dir = os.getenv("IMG_DIR", "test/in")
        if not os.path.isdir(img_dir):
            print(f"[Locust] IMG_DIR not found: {img_dir}")
            raise StopUser()

        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
        paths = []
        for name in os.listdir(img_dir):
            if name.lower().endswith(exts):
                paths.append(os.path.join(img_dir, name))
        if not paths:
            print(f"[Locust] No images found in {img_dir}")
            raise StopUser()
        self._paths = paths
        print(f"[Locust] Loaded {len(paths)} images from {img_dir}")

    def on_start(self):
        self._load_paths()

    @task
    def mask(self):
        paths = getattr(self, "_paths", None)
        if not paths:
            self._load_paths()
        path = random.choice(self._paths)
        with open(path, "rb") as f:
            with self.client.post(
                "/mask?mask_rgba=true",
                files={"file": f},
                timeout=120,
                catch_response=True,
                name="/mask",
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"{resp.status_code}: {resp.text[:200]}")
                else:
                    resp.success()
