import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 解析 JSON 为 Python 字典
    return data

data = load_json('D:/Zero-shot-ref-seg/Data/anns/refcoco/testB.json')

modified_query = load_json('Data/output_cvt_test_B.json')


cvt_dict = {}
for pair in modified_query:
    cvt_dict[pair['origin']] = pair['convert']


newdata = []

for each in data:
    sentences = each['sentences']
    longest_text = max(sentences, key=lambda x: len(x["sent"]))["sent"]

    onedata = {
        "img_name": each['img_name'],
        "origin_query": longest_text,
        "converted": cvt_dict[longest_text],
        "groundtruth_bbox": each['bbox'],
        "segment_id": each['segment_id']
    }

    newdata.append(onedata)


print(len(newdata))

# 保存为 JSON 文件
with open("modified_dataset_B.json", "w", encoding="utf-8") as f:
    json.dump(newdata, f, ensure_ascii=False, indent=4)

    