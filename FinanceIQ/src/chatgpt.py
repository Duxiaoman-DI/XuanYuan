import os
import argparse
import numpy as np
import openai
import tiktoken
from tqdm import tqdm
from time import sleep
from utils import choices, format_example, gen_prompt, run_eval


openai.api_key = "YOUR_API_KEY"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def get_response(inputs):
    completion = None
    retry = 0
    while completion is None:
        try:
            completion = openai.ChatCompletion.create(
                engine="gpt-3.5-turbo",
                messages=[{"role": "user", "content": inputs}]
                )
            return completion.choices[0].message['content']
        except Exception as msg:
            print(msg)
            retry += 1
            if retry > 3:
                return 'ChatGPT failed'

def eval(subject, dev_df, test_df, num_few_shot, max_length, cot, **kwargs):
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0])):
        try:
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            prompt = gen_prompt(dev_df, subject, prompt_end, num_few_shot, encoding, max_length, cot=cot)
            label = test_df.iloc[i, test_df.shape[1] - 1]
        except Exception as e:
            print(f'failed to format example {i}')
            continue

        pred = get_response(prompt)
        all_preds.append(pred.replace("\n", "") if pred is not None else "")
        if pred and pred[0] in choices:
            cors.append(pred[0] == label)

    acc = np.mean(cors)
    print(f"Average accuracy {acc:.3f} - {subject}")
    print(f"{len(cors)} results, {len(all_preds)-len(cors)} inappropriate formated answers.")
    return all_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="../data")
    parser.add_argument("--save_dir", "-s", type=str, default="../results/ChatGPT")
    parser.add_argument("--num_few_shot", "-n", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()
    print('\n\n\n=========================')
    print(f'args = {args}')
    print('=========================')

    run_eval(None, None, eval, args)

