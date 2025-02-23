import sys
sys.path.append(r'D:/Zero-shot-ref-seg/Img2Cap')
sys.path.append(r'D:/Zero-shot-ref-seg/ObjDetect')
sys.path.append(r'D:/Zero-shot-ref-seg/Reasoner')

from Img2Cap.LMM_API import getCaptionFromLMM
from ObjDetect.BoxGen import getBoxFromText
from Reasoner.LLM_API_calling import modify_query
from Seg.GenSeg import getSegFromBox
import ast


image_path = r"./Data/images/dogs.jpg"

total_caption = getCaptionFromLMM(image_path)

print(f"Image caption:\n{total_caption}")

query = 'The yellow dog on the left'

modified = modify_query(query, total_caption)

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


segs = getSegFromBox(image_path, boxes, True)