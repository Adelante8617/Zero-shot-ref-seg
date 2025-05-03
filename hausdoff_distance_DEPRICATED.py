import numpy as np
from scipy.spatial.distance import cdist

def hausdorff_distance(mask1, mask2, percentile=95):
    """
    计算两个二值掩码之间的 Hausdorff 距离（使用给定百分位降低离群点影响）

    参数:
        mask1, mask2: numpy 数组，二值掩码
        percentile: 距离的百分位（默认取95%）

    返回:
        Hausdorff 距离
    """
    # 获取掩码中非零点的坐标
    points1 = np.argwhere(mask1 > 0)
    points2 = np.argwhere(mask2 > 0)

    if len(points1) == 0 or len(points2) == 0:
        return 1e6

    # 计算距离矩阵（两两点之间欧氏距离）
    dists_1_to_2 = cdist(points1, points2)
    dists_2_to_1 = cdist(points2, points1)

    # 对每个点，取最小距离
    min_dists_1_to_2 = np.min(dists_1_to_2, axis=1)
    min_dists_2_to_1 = np.min(dists_2_to_1, axis=1)

    # 取前 percentile% 最大值作为近似 Hausdorff 距离
    hd_1_to_2 = np.percentile(min_dists_1_to_2, percentile)
    hd_2_to_1 = np.percentile(min_dists_2_to_1, percentile)

    return max(hd_1_to_2, hd_2_to_1)

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

def compute_overall_hausdoff_dist(ground_truth_filepath="./Data/masks/refcoco/", json_paths=['output_seg_api.jsonl'], filename_prefix="Outputs/OutputMasks/SegData_2/"):
    
    def preprocess_mask(image_path):
        """
        读取并预处理 mask 图像，返回二值图像
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        _, binary_mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)  # 二值化，范围变成 0 和 1
        return binary_mask
    
    all_result_to_process = []

    hd_list = []

    for json_path in json_paths:
    
        result = load_jsonl(json_path)

        all_result_to_process.extend(result)



    for data in tqdm(all_result_to_process):
        seg_id = data["segment_id"]
        
        filename = filename_prefix + f"{seg_id}_seg.png"
        origin_file_path = ground_truth_filepath + filename.replace('_seg', '').replace(filename_prefix, '')
        result_file_path = filename


        mask1 = preprocess_mask(origin_file_path)
        mask2 = preprocess_mask(result_file_path)

        hd = hausdorff_distance(mask1, mask2)

        hd_list.append(hd)

        

    average_hd = np.mean(hd_list)
    return average_hd

if __name__ == "__main__":
    compute_overall_hausdoff_dist(json_paths=['output_seg_test_B_local.jsonl'],filename_prefix="Seg_Test_B_Local/")