###
```
python -m pip install locust
pip install -r requirements.txt
python remove_bg.py \
  --ckpt weights/BiRefNet-general-epoch_244.pth \
  --input test/in \
  --output test/out \
  --output_mask test/out-masks \
  --alpha_power 1.4 \
  --edge_refine \
  --max_side 1024 \
  --per_image
python fastapi_app.py --ckpt weights/BiRefNet-general-epoch_244.pth --host 0.0.0.0 --port 8000
IMG_DIR=/workspace/birefnet/test/in locust -f locustfile.py --headless -u 16 -r 2 -t 1m --host http://localhost:8000
```