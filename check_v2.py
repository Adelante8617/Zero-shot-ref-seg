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


v2_res = load_jsonl('output_v2prompt_test_B_cvt.jsonl')

emp_cnt = 0

def iou(box1, box2):
    # 解析坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集坐标
    x_inter1 = max(x1_1, x1_2)
    y_inter1 = max(y1_1, y1_2)
    x_inter2 = min(x2_1, x2_2)
    y_inter2 = min(y2_1, y2_2)
    
    # 计算交集的宽度和高度
    inter_width = max(0, x_inter2 - x_inter1)
    inter_height = max(0, y_inter2 - y_inter1)
    
    # 计算交集面积
    inter_area = inter_width * inter_height
    
    # 计算各自的面积
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算 IoU
    iou = inter_area / (area_1 + area_2 - inter_area) if (area_1 + area_2 - inter_area) != 0 else 0
    
    return iou



mIOU = 0

for line in v2_res:
    if line['gen_box'] == []:
        emp_cnt += 1

    else:
        maxval = 0
        for box in line['gen_box']:
            maxval = max(maxval, iou(box, line['groundtruth_bbox']))
        mIOU += maxval

print(emp_cnt, '/', len(v2_res) ,'=', emp_cnt/len(v2_res)*100, '%')

print(mIOU/(len(v2_res) ))