import requests
import ast

url = "https://api.siliconflow.cn/v1/chat/completions"

base64_image = ""

import base64

def image_to_base64(image_path):
    # 打开图片文件
    with open(image_path, "rb") as image_file:
        # 将图片内容编码为base64
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# 示例
image_path = r"../Data/images/dogs.jpg"  # 替换为你自己的图片路径
base64_image = image_to_base64(image_path)


import json  
from openai import OpenAI

client = OpenAI(
    api_key="sk-zszeipcwnpjtuksmfqwttkgnivzfawfuhqhzbqzaafaakltx", # 从https://cloud.siliconflow.cn/account/ak获取
    base_url="https://api.siliconflow.cn/v1"
)

response = client.chat.completions.create(
        model="Pro/Qwen/Qwen2-VL-7B-Instruct",
        
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        #"url": f"data:image/jpeg;base64,{base64_image}",
                        'url':'https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png',
                        "detail":"low"
                    }
                },
                {
                    "type": "text",
                    "text": "尽可能详细地描述这张图片中的所有物品.不要漏掉值得注意的细节"
                }
            ]
        }],
        max_tokens= 1024,
        stream=True,
        
)

for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message, end='', flush=True)


