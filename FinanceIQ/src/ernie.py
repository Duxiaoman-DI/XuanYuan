import argparse
import numpy as np
from tqdm import tqdm
from time import sleep
import requests
import json

from utils import choices, format_example, gen_prompt, run_eval

API_KEY = "YOUR-API-KEY"
SECRET_KEY = "YOUR-SECRET-KEY"

def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def get_response(inputs):
    completion = None
    retry = 0
    while completion is None:
        try:
            message = {"messages": [{"role": "user","content": inputs}]}
            payload = json.dumps(message)
            headers = {'Content-Type': 'application/json'}
            response = requests.request("POST", url, headers=headers, data=payload)
            return response.json()['result']
        except Exception as msg:
            print(msg)
            retry += 1
            if retry > 3:
                return '生成失败'
            sleep(3)
            continue

def eval(subject, dev_df, test_df, num_few_shot, max_length, cot, **kwargs):
    cors = []
    all_preds = []

    for i in tqdm(range(test_df.shape[0])):
        try:
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            prompt = gen_prompt(dev_df, subject, prompt_end, num_few_shot, None, max_length, cot=cot)
            label = test_df.iloc[i, test_df.shape[1] - 1]
        except Exception as e:
            print(f'failed to format example {i}: {e}')
            continue

        pred = get_response(prompt)
        if pred and pred[0] in choices:
            cors.append(pred[0] == label)
        all_preds.append(pred.replace("\n", "") if pred is not None else "")

    acc = np.mean(cors)
    print(f"Average accuracy {acc:.3f} - {subject}")
    print(f"{len(cors)} results, {len(all_preds)-len(cors)} inappropriate formated answers.")
    return all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="../data_modify")
    parser.add_argument("--save_dir", "-s", type=str, default="../results_modify/ErnieBot-turbo")
    parser.add_argument("--num_few_shot", "-n", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--cot", action='store_true')
    parser.add_argument("--version", "-v", type=str, default="ERNIE-Bot-turbo")
    args = parser.parse_args()
    print('\n\n\n=========================')
    print(f'args = {args}')
    print('=========================')

    token = get_access_token()
    if args.version == "ERNIE-Bot-turbo":
        print('Using ERNIE-Bot-turbo') 
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={token}"
    else:
        print('Using ERNIE-Bot')
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={token}"

    run_eval(None, None, eval, args)
