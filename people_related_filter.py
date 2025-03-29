import requests
import ast

from tqdm import tqdm
import json

def load_jsonl(file_path):
    data = []
    """逐行读取 JSONL 文件，返回生成器"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def getAPIKEY():
    with open(r'D:/Zero-shot-ref-seg/api-keys.txt', 'r') as f:
        API_KEY = f.read()
    return API_KEY

def one_message(input_text):

    API_KEY = getAPIKEY()

    prompt = '''
I will give you a sentence, which is a description to an object. You need to judge if this sentence is pointed to a person.
For example:
- "The guy in white shirt" -> is human
- "The pants wearing on that boy" -> about pants, not human
- "The girl in the middle of two boy" -> is human
- "The man's leg" -> about a part of human, not a person

The input sentence is probably being vague. If you feel puzzled, return "puzzled: True" in given format. If you can understand it without doubts, return False in this term.

You need to judge, but don't need any explaining. Your response should follow the format:
{
    "is_human":True/False, 
    "puzzled": True/False
}
The given sentence is:

'''

    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "deepseek-ai/DeepSeek-V2.5",
        "messages": [
            {
                "role":"system",
                "content":prompt,
            },
            {
                "role": "user",
                "content": input_text
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
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    try:
        dict_obj = ast.literal_eval(response.text)
        #print(dict_obj)
    except:
        return ""

    return dict_obj['choices'][0]['message']['content']

newdataset = []

not_relate_with_human = []

if __name__ == "__main__111":
    dataset = load_jsonl("output_human_judging.jsonl")
    cnt = 0
    for data in dataset:
        try:
            cnt += 1 if data['is_human'] == True else 0

            if not data['is_human']:
                not_relate_with_human.append(data)
        except:
            print(data['rawtext'])

    print(cnt/len(dataset))

    with open("output_not_human.jsonl", "a", encoding="utf-8") as f:
        for new_line in not_relate_with_human:
            json.dump(new_line, f, ensure_ascii=False)
            f.write("\n")  # 每个 JSON 对象独占一行

if __name__ == "__main__":
    dataset = load_jsonl("output_human_judging.jsonl")
    cnt = 0
    for data in dataset:
        try:
            cnt += 1 if data['puzzled'] == True else 0

        except:
            print(data['rawtext'])

    print(cnt/len(dataset), cnt)

    

