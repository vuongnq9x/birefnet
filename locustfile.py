import os
import random

from locust import HttpUser, task, between


class MaskUser(HttpUser):
    wait_time = between(0, 0)

    @task
    def mask(self):
        img_dir = os.getenv("IMG_DIR", "test/in")
        paths = getattr(self, "_paths", None)
        if not paths:
            exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
            paths = []
            for name in os.listdir(img_dir):
                if name.lower().endswith(exts):
                    paths.append(os.path.join(img_dir, name))
            if not paths:
                raise RuntimeError(f"No images found in {img_dir}")
            self._paths = paths

        path = random.choice(self._paths)
        with open(path, "rb") as f:
            self.client.post(
                "/mask?mask_rgba=true",
                files={"file": f},
                timeout=120,
            )
