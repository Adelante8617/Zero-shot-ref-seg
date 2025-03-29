import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 解析 JSON 为 Python 字典
    return data

data = load_json('D:/Zero-shot-ref-seg/Data/anns/refcoco/train.json')

cat = []

for line in data:
    cat.append(line['cat'])

for c in set(cat):
    print("cat:", c, "count:", cat.count(c))