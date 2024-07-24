import sys
from typing import Union
sys.path.append('../mmdetection/')

import os
import json
import mmcv
import torch
import numpy as np
from tqdm import trange
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer

def renderResult(model, image: np.ndarray, result, pred_score_thr: float = 0.3, show: bool = True, out_file: Union[str, None] = None) -> bool:
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta

    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        pred_score_thr=pred_score_thr,
        out_file=out_file,
    )

    if show:
        visualizer.show()

    return True

config_file = '/home/chli/github/XRay/mm-detection/mm_detection/Config/co_detr_xray_v1.py'
checkpoint_file = '/home/chli/github/XRay/mm-detection/output/co_detr/xray-v1.pth'
test_image_folder_path = '/home/chli/Dataset/X-Ray/test1/'
save_folder_name = 'co-detr-train4k-thr01'
score_threshold = 0.01
device = 'cuda:0'
render = False

save_folder_path = './output/' + save_folder_name
os.makedirs(save_folder_path + '/', exist_ok=True)

register_all_modules()
model = init_detector(config_file, checkpoint_file, device=device)

image_filename_list = os.listdir(test_image_folder_path)

image_filename_list.sort()

results_list = []

with torch.no_grad():
    for i in trange(len(image_filename_list)):
        image_filename = image_filename_list[i]
        if image_filename[-4:] != '.jpg':
            continue

        current_results = []

        image_file_path = test_image_folder_path + image_filename

        image = mmcv.imread(image_file_path, channel_order='bgr')

        result = inference_detector(model, image)

        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.detach().clone().cpu().numpy()
        labels = pred_instances.labels.detach().clone().cpu().numpy()
        scores = pred_instances.scores.detach().clone().cpu().numpy()

        renderResult(model, image, result, score_threshold, render, save_folder_path + '/' + str(i) + '.jpg')

        valid_bbox_mask = scores >= score_threshold

        for i in range(8):
            current_label_mask = (labels == i) & valid_bbox_mask

            current_label_bboxes = bboxes[current_label_mask]
            current_label_scores = scores[current_label_mask]

            current_label_results = np.hstack([current_label_bboxes, current_label_scores.reshape(-1, 1)])

            current_results.append(current_label_results.tolist())

        results_list.append(current_results)

    with open(save_folder_path + '.json', 'w') as f:
        json.dump(results_list, f)
