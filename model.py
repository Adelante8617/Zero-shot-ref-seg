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


image_path = r"./Data/images/dogs.jpg"

total_caption = getCaptionFromLMM(image_path)

print(f"Image caption:\n{total_caption}")

query = 'A green bird'

modified = modify_query(query)

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

selected_ids = select_from_list(total_caption, modified, caption_list)

start, end = selected_ids.find('{'), selected_ids.rfind('}')

selected_dict = None


if start != -1 and end != -1 and start < end:
    result = selected_ids[start:end+1]
    selected_dict = ast.literal_eval(result)

item_index = selected_dict['selected_ids']

print(item_index)

selected_boxes = [boxes[i-1] for i in item_index if i-1<len(boxes)]

segs = getSegFromBox(image_path, selected_boxes, True)
