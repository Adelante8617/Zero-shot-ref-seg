import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import warnings
warnings.filterwarnings("ignore", message=".*MultiScaleDeformableAttention.*")

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = "D:/Zero-shot-ref-seg/Data/images/dogs.jpg"
image = Image.open(image_path)
# Check for cats and remote controls
text_labels = [["dog"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

result = results[0]

boxlist = []

for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
    boxlist.append(box)


from PIL import Image, ImageDraw

image = Image.open(image_path).convert("RGB")

# 要绘制的两个框，格式为 (x1, y1, x2, y2)


# 创建一个可绘图对象
draw = ImageDraw.Draw(image)

# 画出框（红色框，线宽为3）
for box in boxlist:
    draw.rectangle(box, outline="red", width=3)

# 保存新图片
output_path = 'output_with_boxes.jpg'
image.save(output_path)

print(f"图片已保存到：{output_path}")
