# python3 train_simple.py \
#   --img 640 \
#   --batch-size 32 \
#   --epochs 20 \
#   --data data/kaist-rgbt.yaml \
#   --cfg models/yolov5n_kaist-rgbt-deep.yaml \
#   --weights /home/yewon/project/AUE8088/runs/train/yolov5n-rgbt-1/weights/best.pt \
#   --workers 16 \
#   --optimizer SGD \
#   --cos-lr \
#   --hyp /home/yewon/project/AUE8088/data/hyps/hyp.finetune.yaml \
#   --name sin \
#   --device 1 \
#   --rgbt

echo "=== Validation: BASELINE(YOLOv5x+mosaic) ==="
python3 val_rgbt.py \
    --weights runs/train/sin/weights/best.pt \
    --data data/kaist-rgbt.yaml \
    --task test \
    --rgbt \
    --name ssin \
    --save-json \
    --device 2

echo "=== submit.py ==="
python3 submit.py \
    --target-runs ssin