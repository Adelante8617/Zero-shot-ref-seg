import sys
sys.path.append(r'D:/Zero-shot-ref-seg/Img2Cap')
sys.path.append(r'D:/Zero-shot-ref-seg/ObjDetect')
sys.path.append(r'D:/Zero-shot-ref-seg/Reasoner')

#from Img2Cap.LMM_API import generate_caption
#from Img2Cap.local_captioner import generate_caption
#from ObjDetect.BoxGen import getBoxFromText
#from Reasoner.Local_LLM_reasoner import modify_query, select_from_list
from Seg.GenSeg import getSegFromBox
import ast
from PIL import Image
import json
import numpy as np
from time import time
from tqdm import tqdm

def load_jsonl(file_path):
    data = []
    """逐行读取 JSONL 文件，返回生成器"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append( json.loads(line))

    return data

dataset = load_jsonl('output_v2prompt_test_B_cvt.jsonl')

new_dataset = []

cnt = 0

for data in tqdm(dataset[:]):
    cnt += 1
    image_path = "./Data/train2014/train2014/"+data['img_name']
    selected_boxes = data['gen_box']
    selected_boxes = np.array(selected_boxes, dtype=int)
    segs = getSegFromBox(image_path, selected_boxes, visualize=False)
    binary_arr = segs
    arr_to_img = np.array(binary_arr, dtype=np.uint8) * 255
    img = Image.fromarray(arr_to_img, mode='L') 
    save_path = './Seg_Test_B_Api_cvt/'+str(data['segment_id'])+"_seg.png"
    img.save(save_path)
    data['savepath'] = save_path
    new_dataset.append(data)
    


with open("output_seg_test_B_api_cvt.jsonl", "a", encoding="utf-8") as f:
    for new_line in new_dataset:
        json.dump(new_line, f, ensure_ascii=False)
        f.write("\n")  # 每个 JSON 对象独占一行
