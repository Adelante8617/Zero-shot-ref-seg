import cv2
import numpy as np
import os
from tqdm import tqdm

def compute_iou(img1, img2):
    """计算单对图像的 IoU"""
    intersection = np.logical_and(img1 == 255, img2 == 255).sum()
    union = np.logical_or(img1 == 255, img2 == 255).sum()
    return intersection / union if union > 0 else 0

def compute_miou(ground_truth_filepath="./Data/masks/refcoco/", mode='API'):
    result_folder = ""
    if mode == 'API':
        result_folder = './SegData_2/'
    elif mode == 'Local':
        result_folder = './SegData_local/'
    else:
        print("UNKNOWN MODE")
        return []
    
    ious = []
    
    for filename in tqdm(os.listdir(result_folder)):
        origin_file_path =  ground_truth_filepath + filename.replace('_seg', '')
        result_file_path = os.path.join(result_folder, filename)
        
        gt_img = cv2.imread(origin_file_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)

        # 确保是二值图（如需要可调整阈值）
        _, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        _, pred_img = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)
        iou = compute_iou(gt_img, pred_img)
        ious.append(iou)

    miou = np.mean(ious)
    return miou



miou_value = compute_miou(mode="API")
print(f"API Calling Mean IoU: {miou_value:.4f}")
