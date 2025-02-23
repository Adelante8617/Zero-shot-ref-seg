import requests
import ast
from Reasoner.prompt_text import prompt_for_modify, prompt_for_verify, prompt_for_select

def getAPIKEY():
    with open(r'D:/Zero-shot-ref-seg/api-keys.txt', 'r') as f:
        API_KEY = f.read()
        print('Get API KEY :', API_KEY)
    return API_KEY

def one_message(input_text, role='modifier'):
    role_dict = {
        'modifier': prompt_for_modify,
        'verifier': prompt_for_verify,
        'selector': prompt_for_select
    }

    API_KEY = getAPIKEY()

    prompt = role_dict[role]

    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
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

    dict_obj = ast.literal_eval(response.text)
    #print(dict_obj)

    return dict_obj['choices'][0]['message']['content']

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
