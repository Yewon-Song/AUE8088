# echo "=== Train: BASELINE(YOLOv5x+mosaic+lossdiff+hyp-high) ==="
# python3 train_simple.py \
#     --img 640 \
#     --batch-size 32 \
#     --epochs 30 \
#     --data data/kaist-rgbt.yaml \
#     --cfg /home/yewon/project/AUE8088/models/yolov5n_kaist-rgbt-deep.yaml\
#     --weights yolov5n.pt \
#     --workers 16 \
#     --hyp /home/yewon/project/AUE8088/data/hyps/hyp.scratch-high-focal.yaml \
#     --rgbt \
#     --device 3 \
#     --name yolov5n-rgbt-base-loss \
# yolov12s_kaist-rgbt.yaml
# CUDA_VISIBLE_DEVICES=3 
# python3 train_simple.py \
#   --img 640 \
#   --batch-size 16 \
#   --epochs 20 \
#   --data data/kaist-rgbt.yaml \
#   --cfg models/yolov5n_kaist-rgbt-deep.yaml \
#   --weights /home/yewon/project/AUE8088/runs/train/hanbal2/weights/best.pt \
#   --workers 16 \
#   --optimizer SGD \
#   --cos-lr \
#   --hyp /home/yewon/project/AUE8088/data/hyps/hyp.finetune.yaml \
#   --name nnom \
#   --device 3 \
#   --rgbt

# echo "=== Validation: BASELINE(YOLOv5x+mosaic) ==="
# python3 val_rgbt.py \
#     --weights runs/train/nnom/weights/best.pt \
#     --data data/kaist-rgbt.yaml \
#     --task test \
#     --rgbt \
#     --name nnom \
#     --save-json \
#     --device 1

echo "=== submit.py ==="
python3 submit.py \
    --target-runs nnom