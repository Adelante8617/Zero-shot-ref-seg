import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 设置图像文件夹路径
image_folder = 'D:/Zero-shot-ref-seg/Data/masks/refcoco'

# 获取所有图片文件名
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]

target_size = (256, 256)

sum_image = np.zeros((target_size[1], target_size[0]), dtype=np.float32)

for file in tqdm(image_files):
    img_path = os.path.join(image_folder, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 调整为相同尺寸
    img_resized = cv2.resize(img, target_size)

    # 累加
    sum_image += img_resized.astype(np.float32)

# 归一化
epsilon = 1e-5
log_sum_image = np.log(sum_image + epsilon)

# 归一化到 0~1 区间
log_sum_norm = cv2.normalize(log_sum_image, None, 0, 1, cv2.NORM_MINMAX)

# 可视化热力图
plt.imshow(log_sum_norm, cmap='plasma')  # 可选：'hot', 'jet', 'magma', 'plasma'
plt.colorbar()
plt.title("Log Heatmap of Overlapping Regions (256x256)")
plt.show()