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

def compute_miou(ground_truth_filepath="./Data/masks/refcoco/", json_path='output_seg_api.jsonl', filename_prefix="Outputs/OutputMasks/SegData_2/"):
    
    result = load_jsonl(json_path)
    
    ious = []

    no_box = 0
    
    for data in tqdm(result):
        if len(data['gen_box']) == 0 :
            no_box += 1
            ious.append(0)
            continue

        seg_id = data["segment_id"]
        filename = filename_prefix + f"{seg_id}_seg.png"
        origin_file_path =  ground_truth_filepath + filename.replace('_seg', '').replace(filename_prefix, '')
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

    return miou, miou * len(result) /(len(result) - no_box) 

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



import math


def compute_ciou(pred_box, gt_box):
    # [x1, y1, x2, y2]
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box

    # 面积交并
    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    # 中心距离
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    d2 = (pcx - gcx)**2 + (pcy - gcy)**2

    # 包含两个框的最小框对角线距离
    cx1 = min(px1, gx1)
    cy1 = min(py1, gy1)
    cx2 = max(px2, gx2)
    cy2 = max(py2, gy2)
    c2 = (cx2 - cx1)**2 + (cy2 - cy1)**2

    D = d2 / c2 if c2 > 0 else 0

    # aspect ratio penalty
    pred_w, pred_h = px2 - px1, py2 - py1
    gt_w, gt_h = gx2 - gx1, gy2 - gy1
    v = (4 / (math.pi**2)) * (math.atan(gt_w / gt_h) - math.atan(pred_w / pred_h))**2

    if 1 - iou + v != 0:
        alpha = v / (1 - iou + v) if iou >= 0.5 else 0
    else:
        #print("divided by 0")
        #print(v, iou)
        alpha = 0
    ciou = iou - D - alpha * v
    return ciou


def compute_overall_ciou(ground_truth_filepath="./Data/masks/refcoco/", json_path='output_seg_api.jsonl', filename_prefix="Outputs/OutputMasks/SegData_2/"):
    result = load_jsonl(json_path)

    total_ciou = 0
    count = 0
    empty_count = 0

    for data in tqdm(result):
        seg_id = data["segment_id"]
        
        filename = filename_prefix + f"{seg_id}_seg.png"
        origin_file_path = ground_truth_filepath + filename.replace('_seg', '').replace(filename_prefix, '')
        result_file_path = filename

        gt_img = cv2.imread(origin_file_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)

        if pred_img is None:
            count += 1
            continue

        _, gt_mask = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        _, pred_mask = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)

        # 得到 bounding box
        gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not gt_contours or not pred_contours:
            empty_count += 1
            continue

        gt_box = cv2.boundingRect(max(gt_contours, key=cv2.contourArea))  # x, y, w, h
        pred_box = cv2.boundingRect(max(pred_contours, key=cv2.contourArea))

        # 转为 x1, y1, x2, y2
        gx1, gy1, gw, gh = gt_box
        px1, py1, pw, ph = pred_box
        gt_box_xyxy = [gx1, gy1, gx1 + gw, gy1 + gh]
        pred_box_xyxy = [px1, py1, px1 + pw, py1 + ph]

        ciou = compute_ciou(pred_box_xyxy, gt_box_xyxy)
        total_ciou += ciou
        count += 1

    all_res = total_ciou / (count+empty_count) if count+empty_count > 0 else 0
    box_res = total_ciou / count if count > 0 else 0

    return all_res, box_res

def compute_overall_pixel_accuracy(ground_truth_filepath="./Data/masks/refcoco/", json_path='output_seg_api.jsonl', filename_prefix="Outputs/OutputMasks/SegData_2/"):
    result = load_jsonl(json_path)

    total_correct = 0
    total_pixels = 0

    for data in tqdm(result):
        seg_id = data["segment_id"]
        
        filename = filename_prefix + f"{seg_id}_seg.png"
        origin_file_path = ground_truth_filepath + filename.replace('_seg', '').replace(filename_prefix, '')
        result_file_path = filename

        gt_img = cv2.imread(origin_file_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)


        _, gt_mask = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        _, pred_mask = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)


        # 像素一致的个数
        correct = (gt_mask == pred_mask).sum()
        total = gt_mask.size

        total_correct += correct
        total_pixels += total

    all_res = total_correct / total_pixels if total_pixels > 0 else 0
    return all_res


json_path='output_seg_baseline_B_origin.jsonl'

filename_prefix="Seg_Baseline_B_origin/"

miou_value = compute_miou(json_path=json_path, filename_prefix=filename_prefix)
print(f"Mean IoU: {miou_value[0]:.4f},  {miou_value[1]:.4f}")
ciou_value = compute_overall_ciou(json_path=json_path, filename_prefix=filename_prefix)
print(f"Mean CIoU: {ciou_value[0]:.4f},  {ciou_value[1]:.4f}")
acc = compute_overall_pixel_accuracy(json_path=json_path, filename_prefix=filename_prefix)
print(f"Accuracy: {acc:.4f}")


