import os
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
from utils import choices, format_example, gen_prompt, run_eval

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig

def eval(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    cors = []
    all_preds = []

    for i in tqdm(range(test_df.shape[0])):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            tokenizer=tokenizer,
                            max_length=max_length)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        with torch.no_grad():
            if is_chat_history:
                pred, history = model.chat(tokenizer, prompt, history=[]) # for ChatGLM and InternLM-Chat
            else:
                if is_chat_model:
                    messages = [{"role": "user", "content": prompt}]
                    pred = model.chat(tokenizer, messages) # for model with model.chat() function, e.g. Baichuan-13B-Chat
                else:
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.1)
                    pred = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        all_preds.append(pred.replace("\n", ""))
        if pred and pred[0] in choices:
            cors.append(pred[0] == label)

    acc = np.mean(cors)
    print(f"Average accuracy {acc:.4f} - {subject}")
    print(f"{len(cors)} results, {len(all_preds)-len(cors)} inappropriate formated answers.")
    return all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../results/not_specified")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--with_conf", action='store_true')
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()
    print('\n\n\n=========================')
    print(f'args = {args}')
    print('=========================')
    
    # is LLaMA series model
    is_llama = 'llama' in args.model_name_or_path.lower() \
                or 'alpaca' in args.model_name_or_path.lower()
    if is_llama:
        is_chat_model = False # The LLaMA family models do not support model.chat() function
    else:
        is_chat_model = 'chat' in args.model_name_or_path.lower()

    is_chat_history = 'chatglm' in args.model_name_or_path.lower() \
        or 'internlm-chat' in args.model_name_or_path.lower() \
        or 'qwen-7b-chat' in args.model_name_or_path.lower()
    is_chatglm = 'chatglm' in args.model_name_or_path.lower()

    print(f'model: {args.model_name_or_path}')
    if is_llama:
        tokenizer_class = LlamaTokenizer
        model_class = LlamaForCausalLM
    elif is_chatglm:
        tokenizer_class = AutoTokenizer
        model_class = AutoModel
    else:
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForCausalLM
    tic = time.time()
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        trust_remote_code=True,
                                        load_in_8bit=args.load_in_8bit,
                                        device_map="auto"
                                        )
    if is_chatglm:
        model = model.half().cuda()  # For ChatGLM
    if is_chat_model and not is_chatglm:
        model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
    print(model.generation_config)
    print(f'loaded model: {args.model_name_or_path}  costtime = {time.time()-tic:2f}s')
    
    if args.lora_weights != "":
        model = PeftModel.from_pretrained(
                        model,
                        args.lora_weights,
                        torch_dtype=torch.float16,
                        )
        
    run_eval(model, tokenizer, eval, args)
