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

from time import time

print("Running...")

t_start = time()

image_path = r"./Data/images/dogs.jpg"

total_caption = generate_caption(image_path=image_path)

t_cap = time()
print("Get caption using:", t_cap-t_start, "seconds.\n\n")

print(f"Image caption:\n{total_caption}")

query = 'a yellow, powerful, dynamic creature which is usually regarded as people\'s pet'

modified = modify_query(query)


t_mod = time()
print("Modify query using:", t_mod-t_cap, "seconds.\n\n")

print(f"Modified query:\n{modified}")


start, end = modified.rfind('{'), modified.rfind('}')

content_dict = None


if start != -1 and end != -1 and start < end:
    result = modified[start:end+1]
    content_dict = ast.literal_eval(result)

item_to_fetch = content_dict['item']

print("All item to fetch:",item_to_fetch)

boxes = getBoxFromText(IMAGE_PATH=image_path, TEXT_PROMPT=item_to_fetch)

t_box = time()
print("Get boxes using:", t_box-t_mod, "seconds.\n\n")

print(boxes)

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

print(caption_list, end='\n=======================\n')

t_cap_list = time()
print("From box to each caption using:", t_cap_list-t_box, "seconds.\n\n")

selected_ids = select_from_list(total_caption, content_dict['description'], caption_list )

t_sel = time()
print("Select target using:",t_sel - t_cap_list, "seconds.\n\n")

import re
import ast

def find_last_integer_list(s):
    # 使用正则表达式匹配列表形式的整数
    lists = re.findall(r'\[([0-9, -]*\d+)\]', s)
    if lists:
        # 将字符串转为列表并返回最后一个
        return ast.literal_eval('[' + lists[-1] + ']')
    return []

item_index = find_last_integer_list(selected_ids)

print(item_index)

selected_boxes = [boxes[i-1] for i in item_index if i-1<len(boxes)]

print("Totally using:", t_sel-t_start, "seconds.\n\n")

#segs = getSegFromBox(image_path, selected_boxes, True)

#t_seg = time()
#print("Get segementation using:", t_seg-t_sel, "seconds.\n\n")

#print("Totally using:", t_seg-t_start, "seconds.\n\n")
