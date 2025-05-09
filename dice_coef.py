import cv2
import torch
import numpy as np
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

def dice_coef(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()

    intersection = (pred * target).sum()
    union = pred.pow(2).sum() + target.pow(2).sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def compute_overall_dice_coef(ground_truth_filepath="./Data/masks/refcoco/", json_path='output_seg_api.jsonl', filename_prefix="Outputs/OutputMasks/SegData_2/", json_path2=None, filename_prefix2=None):
    dice_list = []
    
    result = load_jsonl(json_path)

    for data in tqdm(result):
        if len(data['gen_box']) == 0 and False:
            continue


        seg_id = data["segment_id"]
        
        filename = filename_prefix + f"{seg_id}_seg.png"
        origin_file_path = ground_truth_filepath + filename.replace('_seg', '').replace(filename_prefix, '')
        result_file_path = filename

        pred_img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(origin_file_path, cv2.IMREAD_GRAYSCALE)

        # 二值化（假设阈值为127）
        _, pred_bin = cv2.threshold(pred_img, 127, 1, cv2.THRESH_BINARY)
        _, target_bin = cv2.threshold(target_img, 127, 1, cv2.THRESH_BINARY)

        # 转为 PyTorch 张量
        pred_tensor = torch.from_numpy(pred_bin)
        target_tensor = torch.from_numpy(target_bin)

        # 计算 Dice 系数
        dice = dice_coef(pred_tensor, target_tensor)

        dice_list.append(dice)

    #------------------------------------------------------------------------------------------
    if json_path2 is not None:
        result2 = load_jsonl(json_path2)

        for data in tqdm(result2):
            if len(data['gen_box']) == 0 and False :
                continue

            seg_id = data["segment_id"]
            
            filename = filename_prefix2 + f"{seg_id}_seg.png"
            origin_file_path = ground_truth_filepath + filename.replace('_seg', '').replace(filename_prefix2, '')
            result_file_path = filename


            pred_img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)
            target_img = cv2.imread(origin_file_path, cv2.IMREAD_GRAYSCALE)

            # 二值化（假设阈值为127）
            _, pred_bin = cv2.threshold(pred_img, 127, 1, cv2.THRESH_BINARY)
            _, target_bin = cv2.threshold(target_img, 127, 1, cv2.THRESH_BINARY)

            # 转为 PyTorch 张量
            pred_tensor = torch.from_numpy(pred_bin)
            target_tensor = torch.from_numpy(target_bin)

            # 计算 Dice 系数
            dice = dice_coef(pred_tensor, target_tensor)

            dice_list.append(dice)


    return np.mean(dice_list)


jsonpath1 = 'output_baseline_test_B_origin.jsonl'
prefix1 = './Seg_Baseline_B_origin/'

jsonpath2 = 'output_baseline_test_A_origin.jsonl'
prefix2 = './Seg_Baseline_A_origin/'


coef = compute_overall_dice_coef(json_path=jsonpath1, filename_prefix=prefix1, json_path2=jsonpath2, filename_prefix2=prefix2)
print(coef)

