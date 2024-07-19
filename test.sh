cd ../mmdetection

python tools/test.py \
  ../mm-detection/mm_detection/Config/Model/co_detr.py \
  work_dirs/co_detr/xray-v1.pth
