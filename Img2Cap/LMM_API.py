import requests
import ast
from Img2Cap.cvt_img_to_base64fmt import image_to_base64
from Img2Cap.prompt_img import prompt_for_caption
import json  

API_KEY = ''

def getAPIKEY():
    with open(r'D:/Zero-shot-ref-seg/api-keys.txt', 'r') as f:
        API_KEY = f.read()
    return API_KEY




BASE_URL = 'https://api.siliconflow.cn/v1/chat/completions'
MODEL = 'Qwen/Qwen2.5-VL-32B-Instruct'

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

    payload = {
        "model": MODEL, # 替换成你的模型
        "messages": [
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
        "max_tokens": 1024,
        "temperature": 0.01
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer "+ getAPIKEY()
    }

    response = requests.post(url=BASE_URL, json=payload, headers=headers) 
    if response.status_code == 200: 
        rawtext =  response.text
        raw_dict = ast.literal_eval(rawtext)
        #print("A vlm response")
        return raw_dict['choices'][0]['message']['content']
    else:
        print(response.status_code)
        print(response.text)
        return "ERROR OCCURED!"


if __name__ == "__main__":
    image_path = r"../Data/train2014/train2014/COCO_train2014_000000581282.jpg"  

    print(generate_caption(image_path=image_path))
    

