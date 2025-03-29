import numpy as np

def compute_ciou(box1, box2):
    """
    计算两个边界框的 Complete IoU（cIoU）
    box1, box2: [x1, y1, x2, y2]
    """
    # 计算 IoU
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    # 计算中心点距离
    center_x1, center_y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    center_x2, center_y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    center_distance = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
    
    # 计算最小包围矩形的对角线长度
    enclosing_x1 = min(box1[0], box2[0])
    enclosing_y1 = min(box1[1], box2[1])
    enclosing_x2 = max(box1[2], box2[2])
    enclosing_y2 = max(box1[3], box2[3])
    c = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
    
    # 计算长宽比一致性惩罚项 v
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    v = (4 / (np.pi ** 2)) * ((np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2)
    
    # 计算 alpha
    alpha = v / ((1 - iou) + v)
    
    # 计算 cIoU
    ciou = iou - (center_distance / c) - alpha * v
    return ciou


def compute_ciou_batch(boxes_pred, boxes_gt):
    """
    计算多个边界框对的 Complete IoU（cIoU）
    boxes_pred, boxes_gt: 形状为 (N, 4) 的 numpy 数组，每行格式为 [x1, y1, x2, y2]
    """
    ious = []
    for box_pred, box_gt in zip(boxes_pred, boxes_gt):
        ious.append(compute_ciou(box_pred, box_gt))
    return np.mean(ious)

import json

def load_jsonl(file_path):
    data = []
    """逐行读取 JSONL 文件，返回生成器"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data

dataset = load_jsonl("Outputs\output_seg.jsonl")

box_gt, box_pred = [], []

for line in dataset:
    if len(line["gen_box"])>0:
        box_gt.append(line["groundtruth_bbox"])
        box_pred.append(line["gen_box"][0])

print(compute_ciou_batch(boxes_gt=box_gt, boxes_pred=box_pred))