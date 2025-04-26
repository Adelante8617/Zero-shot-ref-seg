import requests
import ast
import gc
from Reasoner.prompt_text import prompt_for_modify, prompt_for_verify, prompt_for_select, prompt_for_modify_v2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

model.to(device)

generation_config = {
        "max_new_tokens": 1024,  # 最大生成长度
        "temperature": 0.01,  # 温度参数
        "do_sample": True,  # 使用采样
        "num_return_sequences": 1,  # 返回的序列数量
        "repetition_penalty": 1.2,  # 重复惩罚
    }

def one_message(input_text, role='modifier'):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    

    torch.cuda.empty_cache()

    role_dict = {
        'modifier': prompt_for_modify_v2,
        'verifier': prompt_for_verify,
        'selector': "This task is easy. DO NOT THINK!!! directly answer the question. "+prompt_for_select
    }


    prompt = role_dict[role]

    text_to_model = prompt + input_text
    inputs = tokenizer(text_to_model, return_tensors="pt").to(device)

    with autocast():
        outputs = model.generate(**inputs, **generation_config)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        #print(msg)
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

def select_from_list(origin_query, total_caption, query, sub_caption_list):
    msg = "\{'general':" + f"'{total_caption}','origin query':'{origin_query}','object_list':'{str(sub_caption_list)}'"+ "\}."+f" The origin query is converted to a more easy understanding version:{query}. But this is a reference information. Always based on the origin query to select."
    msg = one_message(input_text=msg, role='selector')
    return msg

if __name__ == "__main__":
    print(modify_query('A kind of animal that usually regarded as peoples pet, and they can be a guardian of our backyard'))
