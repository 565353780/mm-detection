import sys
sys.path.append('../mmdetection/')

import os
import json
import mmcv
import torch
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer

def renderResult(model, image: np.ndarray, result) -> bool:
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta

    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
    )

    visualizer.show()

    return True

config_file = '/home/chli/github/XRay/mm-detection/mm_detection/Config/co_detr_xray_v1.py'
checkpoint_file = '/home/chli/github/XRay/mm-detection/output/co_detr/xray-v1.pth'
test_image_folder_path = '/home/chli/Dataset/X-Ray/test1/'
device = 'cuda:0'
render = False

register_all_modules()
model = init_detector(config_file, checkpoint_file, device=device)


with torch.no_grad():
    image_filename_list = os.listdir(test_image_folder_path)

    image_filename_list.sort()

    results_list = []

    for image_filename in tqdm(image_filename_list):
        if image_filename[-4:] != '.jpg':
            continue

        current_results = []

        image_file_path = test_image_folder_path + image_filename

        image = mmcv.imread(image_file_path, channel_order='rgb')

        result = inference_detector(model, image)

        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.detach().clone().cpu().numpy()
        labels = pred_instances.labels.detach().clone().cpu().numpy()
        scores = pred_instances.scores.detach().clone().cpu().numpy()

        if render:
            renderResult(model, image, result)

        valid_bbox_mask = scores >= 0.3

        for i in range(8):
            current_label_mask = (labels == i) & valid_bbox_mask

            current_label_bboxes = bboxes[current_label_mask]
            current_label_scores = scores[current_label_mask]

            current_label_results = np.hstack([current_label_bboxes, current_label_scores.reshape(-1, 1)])

            current_results.append(current_label_bboxes.tolist())

        results_list.append(current_results)

    os.makedirs('./output/', exist_ok=True)

    with open('./output/co-detr-xray-v1-train4k-rgb-no_scale.json', 'w') as f:
        json.dump(results_list, f)
