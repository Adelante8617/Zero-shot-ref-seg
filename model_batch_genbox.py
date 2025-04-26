import sys
sys.path.append(r'D:/Zero-shot-ref-seg/Img2Cap')
sys.path.append(r'D:/Zero-shot-ref-seg/ObjDetect')
sys.path.append(r'D:/Zero-shot-ref-seg/Reasoner')

from Img2Cap.LMM_API import generate_caption
#from Img2Cap.local_captioner import generate_caption
from ObjDetect.BoxGen import getBoxFromText
from Reasoner.LLM_API_calling import modify_query, select_from_list
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

all_data = load_jsonl('output_test_B_cvt_v1.jsonl')


gen_box_result = []


for eachdata in tqdm(all_data[509:]):
    image_path = './Data/train2014/train2014/' + eachdata['img_name']
    #print(image_path)

    query = eachdata['converted']

    #modified = modify_query(query,  version = "v2")

    #print(modified)

    total_caption = generate_caption(image_path=image_path)

    #print(total_caption)
    #print()

    if not True:
        try:
            start, end = modified.rfind('{'), modified.rfind('}')

            content_dict = None

            if start != -1 and end != -1 and start < end:
                result = modified[start:end+1]
                content_dict = ast.literal_eval(result)

            item_to_fetch = content_dict['item']
            description = content_dict['description']

        except:
            item_to_fetch = modified
            description = modified

        #print(item_to_fetch)
        eachdata['item_to_fetch'] = item_to_fetch
        eachdata['description'] = description

        with open("output_test_A_cvt_v2.jsonl", "a", encoding="utf-8") as f:
            json.dump(eachdata, f, ensure_ascii=False)
            f.write("\n")  # 每个 JSON 对象独占一行

    item_to_fetch = eachdata['item_to_fetch']
    description = eachdata['description'] 
    boxes = getBoxFromText(IMAGE_PATH=image_path, TEXT_PROMPT=item_to_fetch)


    image = Image.open(image_path)
    caption_list = []

    # 遍历所有矩形框
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # 截取矩形框内的图像
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image_path = f'./Data/cropped_imgs/cropped_image_{i}.jpg'
        cropped_image.save(cropped_image_path)
        sub_caption = generate_caption(cropped_image_path)
        caption_list.append(sub_caption)



    selected_ids = select_from_list(origin_query=query, total_caption=total_caption, query=description, sub_caption_list=caption_list )

    #print(selected_ids)


    item_index = find_last_integer_list(selected_ids)


    selected_boxes = [boxes[i-1].tolist() for i in item_index if i-1<len(boxes)]

    #print(selected_boxes)
    
    eachdata['gen_box'] = selected_boxes
    

    with open("output_test_B_v1_boxes.jsonl", "a", encoding="utf-8") as f:
        json.dump(eachdata, f, ensure_ascii=False)
        f.write("\n")  # 每个 JSON 对象独占一行



