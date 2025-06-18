echo "=== Validation: BASELINE(YOLOv5x+mosaic) ==="
python3 val_rgbt.py \
    --weights runs/train/hanbal2/weights/best.pt \
    --data data/kaist-rgbt.yaml \
    --task test \
    --rgbt \
    --name sinbalnom\
    --save-json \
    --device 2

echo "=== submit.py ==="
python3 submit.py \
    --target-runs sinbalnom