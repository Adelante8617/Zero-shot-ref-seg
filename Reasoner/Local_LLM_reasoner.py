import requests
import ast
#from Reasoner.
from prompt_text import prompt_for_modify, prompt_for_verify, prompt_for_select
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

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

def one_message(input_text, role='modifier'):
    role_dict = {
        'modifier': prompt_for_modify,
        'verifier': prompt_for_verify,
        'selector': prompt_for_select
    }


    prompt = role_dict[role]

    text_to_model = prompt + input_text
    inputs = tokenizer(text_to_model, return_tensors="pt").to(device)

    with autocast():
        outputs = model.generate(**inputs, **generation_config)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text)

    return output_text

def modify_query(query:str,background=""):
    
    pass_state = False
    role_list = ['modifier','verifier', 'selector']

    cur_role_id = 0
    final_query = ''
    max_try = 3
    cur_try = 0
    #msg = "The background is :"+background+'\nYou need to find the following object:' + query
    msg = 'You need to find the following object:' + query

    while not pass_state and cur_try < max_try:
        # current role
        role = role_list[cur_role_id]
        # send message
        msg = one_message(input_text=msg, role=role)
        print(msg)
        #print('============================\n\n')
        # end modify state
        if role == 'modifier':
            # convert to verifier
            cur_role_id = 1
            # save current query
            final_query = msg

        # end verify state and passed
        elif role == 'verifier' and '\'pass\':True' in msg.replace(' ','').replace('\"','\'').replace('\n',''):
            pass_state = True

        # not pass
        elif role == 'verifier' and '\'pass\':True' not in msg.replace(' ','').replace('\"','\'').replace('\n',''):
            cur_role_id = 0
            cur_try += 1
            msg = 'This is the text you need to modify:\n{final_query},\nIt did not pass the checker. The reject reason and revise suggestion is as follow:\n' + msg

        
    return final_query

def select_from_list(total_caption, query, sub_caption_list):
    msg = "\{'general':" + f"'{total_caption}','query':'{query}','object_list':'{str(sub_caption_list)}'"+ "\}"
    msg = one_message(input_text=msg, role='selector')
    print(msg)
    return msg

if __name__ == "__main__":
    print(modify_query('A pipe that can use to suck soft drinks'))
