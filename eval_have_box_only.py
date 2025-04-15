import cv2
import numpy as np
import os
from tqdm import tqdm
import json

def load_jsonl(file_path):
    data = []
    """逐行读取 JSONL 文件，返回生成器"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def compute_iou(img1, img2):
    """计算单对图像的 IoU"""
    intersection = np.logical_and(img1 == 255, img2 == 255).sum()
    union = np.logical_or(img1 == 255, img2 == 255).sum()
    return intersection / union if union > 0 else 0

def compute_miou(ground_truth_filepath="./Data/masks/refcoco/", mode='API'):
    
    result = load_jsonl('output_seg_test_B_api_cvt.jsonl')
    
    ious = []

    no_box = 0
    
    for data in tqdm(result):
        if len(data['gen_box']) == 0 :
            no_box += 1
            continue

        seg_id = data["segment_id"]
        filename = f"Seg_Test_B_Api_cvt/{seg_id}_seg.png"
        origin_file_path =  ground_truth_filepath + filename.replace('_seg', '').replace('Seg_Test_B_Api_cvt/', '')
        result_file_path = filename
        
        gt_img = cv2.imread(origin_file_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)

        # 确保是二值图（如需要可调整阈值）
        _, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        _, pred_img = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)
        iou = compute_iou(gt_img, pred_img)
        ious.append(iou)

    miou = np.mean(ious) 
    print("No box:", no_box, " ratio:", no_box/len(result))
    print("havebox:", len(result) - no_box)

    return miou

def compute_overall_iou(ground_truth_filepath="./Data/masks/refcoco/", mode='API'):
    
    result = load_jsonl('output_seg_api.jsonl')

    total_intersection = 0
    total_union = 0
    
    for data in tqdm(result):
        if len(data['gen_box']) == 0 and False:
            continue
        seg_id = data["segment_id"]
        filename = f"Outputs/OutputMasks/SegData/{seg_id}_seg.png"
        origin_file_path =  ground_truth_filepath + filename.replace('_seg', '').replace('Outputs/OutputMasks/SegData/', '')
        result_file_path = filename
        
        gt_img = cv2.imread(origin_file_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)

        # 确保是二值图（如需要可调整阈值）
        _, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        _, pred_img = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)

        intersection = np.logical_and(gt_img == 255, pred_img == 255).sum()
        union = np.logical_or(gt_img == 255, pred_img == 255).sum()

        total_intersection += intersection
        total_union += union

        

    overall_iou = total_intersection / total_union if total_union > 0 else 0
    return overall_iou



miou_value = compute_miou(mode="API")
print(f"API Calling Mean IoU: {miou_value:.4f}")


