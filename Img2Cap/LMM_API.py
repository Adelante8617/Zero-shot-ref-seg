import requests
import ast
from image_process.cvt_img_to_caption import image_to_base64
from prompt import prompt_for_caption
import json  
from openai import OpenAI

API_KEY = ''
BASE_URL = 'https://api.siliconflow.cn/v1'
MODEL = 'Pro/Qwen/Qwen2-VL-7B-Instruct'

def getCaptionFromLMM(image_path,img_url=None, upload_mode='base64', detail="low"):
    img_url = ""
    if upload_mode == 'base64':
        base64_image = image_to_base64(image_path)
        img_url = 'data:image/jpeg;base64,{base64_image}'
    elif upload_mode == 'web':
        pass
    else:
        print("Unknown mode.")
        return 

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
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
                            'url':img_url,
                            "detail":detail
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_for_caption
                    }
                ]
            }],
            max_tokens= 1024,
            stream=False,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    image_path = r"../Data/images/dogs.jpg"  
    base64_image = image_to_base64(image_path)

    client = OpenAI(
        api_key="",
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
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            #'url':'https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png',
                            "detail":"low"
                        }
                    },
                    {
                        "type": "text",
                        "text": "尽可能详细地描述这张图片中的所有物品.不要漏掉值得注意的细节,特别注重对狗的细节描述，只需作客观的描述，无需形容性的表达"
                    }
                ]
            }],
            max_tokens= 1024,
            stream=False,
            
    )
    print(response.choices[0].message.content)
    


