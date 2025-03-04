import requests
import ast
from Img2Cap.cvt_img_to_base64fmt import image_to_base64
from Img2Cap.prompt_img import prompt_for_caption
import json  
from openai import OpenAI

API_KEY = ''

def getAPIKEY():
    with open(r'D:/Zero-shot-ref-seg/api-keys.txt', 'r') as f:
        API_KEY = f.read()
        print('Get API KEY :', API_KEY)
    return API_KEY




BASE_URL = 'https://api.siliconflow.cn/v1'
MODEL = 'Pro/Qwen/Qwen2-VL-7B-Instruct'

def generate_caption(image_path,img_url=None, upload_mode='base64', detail="low"):
    img_url = ""
    if upload_mode == 'base64':
        base64_image = image_to_base64(image_path)
        img_url = f'data:image/jpeg;base64,{base64_image}'
    elif upload_mode == 'web':
        pass
    else:
        print("Unknown mode.")
        return 

    client = OpenAI(
        api_key=getAPIKEY(),
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
                            "detail":"high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe all objects in this photo **as detailed as possible**."
                    }
                ]
            }],
            max_tokens= 1024,
            stream=False,
            
    )
    #print(response.choices[0].message.content)
    


