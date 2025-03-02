import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

# 启用混合精度


# 加载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

generation_config = {
        "max_length": 8192,  # 最大生成长度
        "temperature": 0.7,  # 温度参数
        "do_sample": True,  # 使用采样
        "num_return_sequences": 1,  # 返回的序列数量
        "repetition_penalty": 1.2,  # 重复惩罚
    }

# 准备输入

while True:
    input_text = "解释我给出的名词，要求不能出现物品本身的名称。例如，当我给出‘雨伞’，你应该回答‘一种雨天使用的可以撑开的长条状工具’。注意，你的回答应该遵循固定的格式如下：\{'input':'雨伞','output':'一种雨天使用的长条状工具'\}。现在，将‘"+input()+"’重述："
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 设置生成参数
    

    # 生成输出
    with autocast():
        outputs = model.generate(**inputs, **generation_config)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 打印输出
    print(output_text)
    print('---------------')