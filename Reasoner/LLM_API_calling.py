import requests
import ast
from prompt import prompt_for_modify

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
        {
            "role":"system",
            "content":prompt_for_modify,
        },
        {
            "role": "user",
            "content": "一种绿色表皮内部红色的适用于夏日解渴"
        }
    ],
    "stream": False,
    "max_tokens": 512,
    "stop": ["null"],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
    
}
headers = {
    "Authorization": "Bearer sk-zszeipcwnpjtuksmfqwttkgnivzfawfuhqhzbqzaafaakltx",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

dict_obj = ast.literal_eval(response.text)

print(dict_obj['choices'][0]['message']['content'])
