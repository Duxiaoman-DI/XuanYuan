import os
import re
import glob
import random
import os.path as osp
import numpy as np
import pandas as pd
import json
from collections import defaultdict
choices = ["A", "B", "C", "D"]

from categories import subjects


def format_example(df, idx, subject, include_answer=True, cot=False):
    question = df.iloc[idx, 0]
    assert isinstance(question, str) and len(question) > 0, 'question is empty'
    prompt = "题目：" + question
    k = df.shape[1] - 2
    for j in range(k):
        option_choice = choices[j] # A, B, C, D
        option_content = df.iloc[idx, j + 1]
        prompt += f"\n{option_choice}. {option_content}"

    # Chain-of-thought
    if cot:
        prompt += "\n逐步分析并给出答案选项。"
    else:
        prompt += "\n答案是："

    if include_answer:
        answer = df.iloc[idx, k + 1]
        prompt += f"{answer}\n\n"
    return prompt

def gen_prompt(dev_df, subject, prompt_end, num_few_shot=0, tokenizer=None, max_length=2048, cot=False):
    if cot: # Chain-of-thought
        prompt = f"以下是关于{subject}的单项选择题，请分析并选出正确答案。\n\n"
    else:
        prompt = f"以下是关于{subject}的单项选择题，请直接给出正确答案的选项。\n\n"

    # If no tokenizer, don't consider max length.
    if tokenizer==None:
        for i in range(num_few_shot):
            example = format_example(dev_df, i, subject)
            prompt += example
        return prompt + prompt_end

    start_end_token_len = len(tokenizer.encode(prompt)+tokenizer.encode(prompt_end))
    # If cannot fit in model even without training data, remove the prompt at the beginning.
    if start_end_token_len>max_length:
        return prompt_end

    prompt_list = []
    if num_few_shot > 0:
        for i in range(num_few_shot):
            example = format_example(dev_df, i, subject)
            prompt_list.append((example, tokenizer.encode(example)))

        while prompt_list != [] and sum(len(e[1]) for e in prompt_list) >= max_length - start_end_token_len:
            print(f"Warning: {len(prompt_list)} shot case exceeds max_input_length, remove 1 shot.")
            longest_length = max([len(e[1]) for e in prompt_list])
            prompt_list = [e for e in prompt_list if len(e[1]) != longest_length]
        for p in prompt_list:
            prompt += p[0]

    return prompt + prompt_end


def run_eval(model, tokenizer, eval, args):

    if model:
        model.eval()

    args.save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for i, subject in enumerate(subjects):
        print(f'Evaluating {i+1}/{len(subjects)} {subject}')
        out_file = os.path.join(args.save_dir, f"results_{subject}.csv")
        if os.path.exists(out_file):  # If result file exist, skip this subject
            print(f'  {subject} exists, skip')
            continue
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + ".csv"), header=0, index_col=0)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + ".csv"), header=0, index_col=0)

        # Call the model to get the answer
        preds = eval(model=model,
                    tokenizer=tokenizer,
                    subject=subject,
                    dev_df=dev_df,
                    test_df=test_df,
                    num_few_shot=args.num_few_shot,
                    max_length=args.max_length,
                    cot=args.cot if 'cot' in args else False)
        test_df['prediction'] = preds

        # Save the generated result as a CSV file
        test_df.to_csv(out_file, header=None)
        print(f'result save to {out_file}')

    # Calculate the accuracy from the generated results
    compute_accuracy(args.save_dir)


def extract_choice(response):
    '''
        Always return a choice, even cannot match by regex,
        to ensure fair comparison to other models.
    '''
    response = str(response)
    if response[0] in choices:
        return response[0]
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])',1),
        (r'([ABCD]) ?选?项(是|为)?正确',1),
        (r'正确的?选项(是|为) ?([ABCD])',2),
        (r'答案(应该)?(是|为)([ABCD])',3),
        (r'选项 ?([ABCD]) ?(是|为)?正确',1),
        (r'选择答案 ?([ABCD])',1),
        (r'答案?：?([ABCD])',1),
        (r'([ABCD])(选?项)?是?符合题意',1),
        (r'答案选项：? ?([ABCD])', 1), # chatglm
        (r'答案(选项)?为(.*?)([ABCD])', 3), # chatgpt

    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 2. Recursive match
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer

    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCD])', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 4. Check the only mentioend choices
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    return choices[random.randint(0,3)]


def compute_accuracy(result_dir=''):

    all_acc = defaultdict(float)
    result = {}
    result['model'] = result_dir.split('/')[-1]
    for subject in subjects:
        try:
            file = glob.glob(osp.join(result_dir, f"results_{subject}.csv"))[0]
        except:
            print(f"Warning, {subject} result file not found")
            continue
        df = pd.read_csv(file, names=['id','question','A','B','C','D','answer','response'], index_col=0)
        if df.iloc[0]['question'] == '1':
            df = df.drop(0)
        df['pred'] = df['response'].apply(extract_choice)
        df['acc'] = df['answer'] == df['pred']
        acc = np.mean(df['acc']) * 100
        all_acc[subject] = acc
        result[subject] = round(acc,2)
    
    for subject in subjects:
        print(f"{subject:40s} {all_acc[subject]:.2f}")
    avg_all_acc = np.mean(list(all_acc.values()))
    print(f"{'Overall':30s} {avg_all_acc:.2f}")

    # Save result as result.json
    result['Overall'] = round(avg_all_acc, 2)
    filename = osp.join(result_dir, 'result.json')
    with open(filename, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        print(f'result save to {filename}')

if __name__ == "__main__":
    pass
