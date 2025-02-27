import sys
sys.path.append(r'D:/Zero-shot-ref-seg/Img2Cap')
sys.path.append(r'D:/Zero-shot-ref-seg/ObjDetect')
sys.path.append(r'D:/Zero-shot-ref-seg/Reasoner')

from Img2Cap.LMM_API import getCaptionFromLMM
from ObjDetect.BoxGen import getBoxFromText
from Reasoner.LLM_API_calling import modify_query, select_from_list
from Seg.GenSeg import getSegFromBox
import ast
from PIL import Image

from time import time


t_start = time()

image_path = r"./Data/images/dogs.jpg"

total_caption = getCaptionFromLMM(image_path)

t_cap = time()
print("Get caption using:", t_cap-t_start, "seconds.\n\n")

print(f"Image caption:\n{total_caption}")

query = 'a yellow, powerful, dynamic creature which is usually regarded as people\'s pet'

modified = modify_query(query)

t_mod = time()
print("Modify query using:", t_mod-t_cap, "seconds.\n\n")

print(f"Modified query:\n{modified}")

modified = modified.replace(' ','')

start, end = modified.find('{'), modified.rfind('}')

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
    sub_caption = getCaptionFromLMM(cropped_image_path)
    caption_list.append(sub_caption)

print(caption_list, end='\n=======================\n')

t_cap_list = time()
print("From box to each caption using:", t_cap_list-t_box, "seconds.\n\n")

selected_ids = select_from_list(total_caption, modified, caption_list)

t_sel = time()
print("Select target using:",t_sel - t_cap_list, "seconds.\n\n")

start, end = selected_ids.find('{'), selected_ids.rfind('}')

selected_dict = None


if start != -1 and end != -1 and start < end:
    result = selected_ids[start:end+1]
    selected_dict = ast.literal_eval(result)

item_index = selected_dict['selected_ids']

print(item_index)

selected_boxes = [boxes[i-1] for i in item_index if i-1<len(boxes)]

segs = getSegFromBox(image_path, selected_boxes, True)

t_seg = time()
print("Get segementation using:", t_seg-t_sel, "seconds.\n\n")

print("Totally using:", t_seg-t_start, "seconds.\n\n")
