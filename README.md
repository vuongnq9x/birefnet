###
```
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
```