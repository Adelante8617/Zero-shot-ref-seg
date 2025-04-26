import sys

import ast
from PIL import Image
from tqdm import tqdm

import re
import ast

import matplotlib.pyplot as plt
from collections import Counter


print("Running...")

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

all_data = load_json('modified_dataset_B.json')

strings_cvt = []
strings_ori = []

cnt = 0

for data in all_data:
    strings_ori.append((data['origin_query']))
    strings_cvt.append((data['converted']))
    if len(data['origin_query']) <= len(data['converted']):
        cnt += 1

all_data = load_json('Outputs/modified_dataset.json')

for data in all_data:
    strings_ori.append((data['origin_query']))
    strings_cvt.append((data['converted']))
    if len(data['origin_query']) <= len(data['converted']):
        cnt += 1

import math

def plot_two_string_length_distributions_binned(list1, list2, save_path, bin_size=10, labels=("Raw data", "Converted")):
    # 统计长度并分桶
    def get_binned_counts(string_list):
        bins = [((len(s) - 1) // bin_size + 1) * bin_size for s in string_list]
        return Counter(bins)

    count1 = get_binned_counts(list1)
    count2 = get_binned_counts(list2)

    # 合并所有可能出现的 bin 上界（排序）
    all_bins = sorted(set(count1.keys()) | set(count2.keys()))
    x_labels = [f"{b - bin_size + 1}-{b}" for b in all_bins]
    x = range(len(all_bins))

    y1 = [count1.get(b, 0) for b in all_bins]
    y2 = [count2.get(b, 0) for b in all_bins]

    bar_width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - bar_width/2 for i in x], y1, width=bar_width, label=labels[0])
    plt.bar([i + bar_width/2 for i in x], y2, width=bar_width, label=labels[1])
    plt.xticks(x, x_labels, rotation=45)
    #plt.xlabel("String Length Range")
    #plt.ylabel("Count")
    #plt.title(f"String Length Distribution Comparison (bin size = {bin_size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

plot_two_string_length_distributions_binned(strings_ori, strings_cvt, './distribution')

def count_mean(ls):
    tmp = 0
    for i in ls:
        tmp+= len(i)

    return tmp / len(ls)

def count_mean_words(ls):
    tmp = 0
    for i in ls:
        tmp+= len(i.split())

    return tmp / len(ls)


print("avg len:", count_mean(strings_ori), count_mean(strings_cvt))
print("avg words:", count_mean_words(strings_ori), count_mean_words(strings_cvt))
print("longer:", cnt)