import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 解析 JSON 为 Python 字典
    return data

def load_jsonl(file_path):
    data = []
    """逐行读取 JSONL 文件，返回生成器"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append( json.loads(line))

    return data

all_data = load_json('Outputs/modified_dataset.json')

all_data.extend(load_json('Outputs/modified_dataset_B.json'))


# 设置图像文件夹路径
image_folder = 'D:/Zero-shot-ref-seg/Data/masks/refcoco'



import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

target_size = (256, 256)

sum_image = np.zeros((target_size[1], target_size[0]), dtype=np.float32)

cnt = 0
emp = 0
for data in tqdm(all_data):

    for kw in ['left','right','between','up','down']:
        if kw in data['origin_query']:
            cnt += 1
            img_path = os.path.join(image_folder, str(data['segment_id'])+".png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 调整为相同尺寸
            img_resized = cv2.resize(img, target_size)

            # 累加
            sum_image += img_resized.astype(np.float32)

            break


# 归一化
epsilon = 1e-5
log_sum_image = (sum_image + epsilon)

# 归一化到 0~1 区间
log_sum_norm = cv2.normalize(log_sum_image, None, 0, 1, cv2.NORM_MINMAX)
plt.axis('off')
# 可视化热力图
plt.imshow(log_sum_norm, cmap='plasma')  # 可选：'hot', 'jet', 'magma', 'plasma'
plt.colorbar()
#plt.title("Heatmap of Overlapping Regions (256x256)")
plt.show()

print(cnt, cnt/len(all_data))