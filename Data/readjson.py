import json
import ast

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 解析 JSON 为 Python 字典
    return data

data = load_json('output_test_B.json')

keys = list(data.keys())

final_contents = []

for key in keys:
    final_contents.append({
        'origin':key,
        'convert':ast.literal_eval(data[key])['choices'][0]['message']['content']
    })

with open("output_cvt_test_B.json", "w", encoding="utf-8") as f:
    json.dump(final_contents, f, ensure_ascii=False, indent=4)