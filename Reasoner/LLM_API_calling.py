import requests
import ast
from prompt import prompt_for_modify, prompt_for_verify

def one_message(input_text, role='modifier'):
    prompt = prompt_for_modify if role == 'modifier' else prompt_for_verify

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
        "Authorization": "Bearer sk-zszeipcwnpjtuksmfqwttkgnivzfawfuhqhzbqzaafaakltx",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    dict_obj = ast.literal_eval(response.text)
    #print(dict_obj)

    return dict_obj['choices'][0]['message']['content']

def modify_query(query:str):
    pass_state = False
    role_list = ['modifier','verifier']
    cur_role_id = 0
    final_query = ''
    max_try = 3
    cur_try = 0
    msg = query

    while not pass_state and cur_try < max_try:
        # current role
        role = role_list[cur_role_id]
        # send message
        msg = one_message(input_text=msg, role=role)
        print(msg)
        print('============================\n\n')
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
            msg = '这是你需要修正的文本:\n{final_query}，\n它并未通过审查，你需要根据如下的审查结果修改：\n' + msg

    
    return final_query

if __name__ == "__main__":
    print(modify_query('一个可以用来回答文本问题的网络程序，它由OpenAI开发'))
