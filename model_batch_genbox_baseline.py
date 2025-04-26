import sys
sys.path.append(r'D:/Zero-shot-ref-seg/Img2Cap')
sys.path.append(r'D:/Zero-shot-ref-seg/ObjDetect')
sys.path.append(r'D:/Zero-shot-ref-seg/Reasoner')

#from Img2Cap.LMM_API import generate_caption
#from Img2Cap.local_captioner import generate_caption
from ObjDetect.BoxGen import getBoxFromText
#from Reasoner.LLM_API_calling import modify_query, select_from_list
#from Seg.GenSeg import getSegFromBox
import ast
from PIL import Image
from tqdm import tqdm

import re
import ast

def find_last_integer_list(s):
    try:
        # 使用正则表达式匹配列表形式的整数
        lists = re.findall(r'\[([0-9, -]*\d+)\]', s)
        if lists:
            # 将字符串转为列表并返回最后一个
            return ast.literal_eval('[' + lists[-1] + ']')
        else:
            return []
    except:
        return []


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

all_data = load_json('Outputs/modified_dataset.json')


gen_box_result = []


for eachdata in tqdm(all_data[:]):
    image_path = './Data/train2014/train2014/' + eachdata['img_name']

    query = eachdata['origin_query']

    boxes = getBoxFromText(IMAGE_PATH=image_path, TEXT_PROMPT=query)

    selected_boxes = [box.tolist() for box in boxes]

    #print(selected_boxes)
    
    eachdata['gen_box'] = selected_boxes
    

    with open("output_baseline_test_A_origin.jsonl", "a", encoding="utf-8") as f:
        json.dump(eachdata, f, ensure_ascii=False)
        f.write("\n")  # 每个 JSON 对象独占一行



