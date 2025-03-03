import requests
from tqdm import tqdm
import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 解析 JSON 为 Python 字典
    return data

data = load_json('anns/refcoco/testA.json')

sentences_to_rewrite = []

for each_pic in data:
    sentences = each_pic['sentences']
    longest_text = max(sentences, key=lambda x: len(x["sent"]))["sent"]
    sentences_to_rewrite.append(longest_text)

print(sentences_to_rewrite[:5],end='============\n\n')

prompt_for_rewrite = '''
Rewrite this sentence. Note that you should convert it into a sentence that have the same meaning, but not so straightforward. For example, you can replace some noun in it with a description to the noun, without the noun itself's appearance.

The sentence to rewrite is:
'''

def getAPIKEY():
    with open(r'D:/Zero-shot-ref-seg/api-keys.txt', 'r') as f:
        API_KEY = f.read()
        print('Get API KEY :', API_KEY)
    return API_KEY

API_KEY = getAPIKEY()



url = "https://api.siliconflow.cn/v1/chat/completions"

rewrited_sentences = []

for sent in tqdm(sentences_to_rewrite):

    payload = {
        "model": "deepseek-ai/DeepSeek-V2.5",
        "messages": [
            {
                "role": "user",
                "content": prompt_for_rewrite + sent
            }
        ],
        "stream": False,
        "max_tokens": 128,
        "stop": None,
        "temperature": 0.01,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    rewrited_sentences.append(response.text)


print(rewrited_sentences[:5],end='============\n\n')


# 组织数据
data = {key: value for key, value in zip(sentences_to_rewrite, rewrited_sentences)}

# 保存为 JSON 文件
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


