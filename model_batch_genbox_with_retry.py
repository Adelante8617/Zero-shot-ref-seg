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

all_data = load_json('./Outputs/modified_dataset_B.json')


gen_box_result = []

def get_item_to_fetch(query, total_caption="", need_background_info=False):
    
    dict_has_content = False


    if need_background_info:
        query = query + "And the description to this scene is:\n" + total_caption

    modified = modify_query(query,version="v2")

    

    try:
        # find {} dict info
        start, end = modified.rfind('{'), modified.rfind('}')

        content_dict = None

        if start != -1 and end != -1 and start < end:
            result = modified[start:end+1]
            # extract this dict
            content_dict = ast.literal_eval(result)

        
        dict_has_content = True

    except:
        content_dict = {}
        content_dict['item'] = modified
        content_dict['description'] = modified

    return content_dict, dict_has_content
    
def one_process(image_path, query):

    total_caption = generate_caption(image_path=image_path)
    
    # first time
    content_dict, dict_has_content = get_item_to_fetch(query=query,  total_caption="", need_background_info=False)

    # retry once
    if (not dict_has_content) or content_dict['item']=="":
        content_dict, _ = get_item_to_fetch(query=query,  total_caption=total_caption, need_background_info=True)

    item_to_fetch = content_dict['item']

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



    selected_ids = select_from_list(origin_query=query, total_caption=total_caption, query=content_dict['description'], sub_caption_list=caption_list)


    item_index = find_last_integer_list(selected_ids)


    selected_boxes = [boxes[i-1].tolist() for i in item_index if i-1<len(boxes)]

    return selected_boxes


for eachdata in tqdm(all_data[:]):
    image_path = './Data/train2014/train2014/' + eachdata['img_name']

    query = eachdata['origin_query']

   
    eachdata['gen_box'] = one_process(query=query, image_path=image_path)

    with open("output_v2prompt_test_B.jsonl", "a", encoding="utf-8") as f:
        json.dump(eachdata, f, ensure_ascii=False)
        f.write("\n")  # 每个 JSON 对象独占一行


