from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

import warnings
warnings.filterwarnings("ignore")

# 加载预训练模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")  # 读取图像
    inputs = processor(images=image, return_tensors="pt")  # 预处理
    output = model.generate(**inputs)  # 生成文本
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption

if __name__=="__main__":
# 示例
    caption = generate_caption("../Data/images/dogs.jpg")
    print("生成的描述:", caption)
