# -*- utf8 -*

import argparse
from conversation import get_conv_template

try:
    from vllm import LLM, SamplingParams
    is_vllm_avaiable = True
    print("use vllm.generate to infer...")
except ImportError:
    from transformers import LlamaForCausalLM, LlamaTokenizer
    is_vllm_avaiable = False
    print("use transformers.generate to infer...")


def infer_vllm(llm, sampling_params, prompt):
    assert llm is not None
    assert sampling_params is not None
    generation = llm.generate(prompt, sampling_params, use_tqdm=False)
    outputs = generation[0].outputs[0].text.strip()
    return outputs


def infer(model, tokenizer, prompt):
    assert model is not None
    assert tokenizer is not None
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p
    )
    outputs = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test XuanYuan-70B-chat with vLLM")
    parser.add_argument("-c", "--checkpoint_path", type=str, help="Checkpoint path")
    parser.add_argument("-n", "--max_new_tokens", type=int, default=1000)
    parser.add_argument("-t", "--temperature", type=float, default=0.95)
    parser.add_argument("-p", "--top_p", type=float, default=0.95)
    args = parser.parse_args()

    llm = None
    sampling_params = None
    model = None
    tokenizer = None

    if is_vllm_avaiable:
        print("loading weight with vLLM...")
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            stop=list(["</s>"]),
            max_tokens=args.max_new_tokens
        )
        llm = LLM(args.checkpoint_path, tensor_parallel_size=8)
    else:
        print("loading weight with transformers ...")
        tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint_path, use_fast=False, legacy=True)
        model = LlamaForCausalLM.from_pretrained(args.checkpoint_path, device_map="auto")

    conv = get_conv_template("XuanYuan-Chat")
    print("########")
    print("输入为: EXIT!! 表示退出")
    print("输入为: CLEAR!! 表示清空上下文")
    print("########")
    while True:
        content = input("输入: ")
        if content.strip() == "EXIT!!":
            print("exit....")
            break
        if content.strip() == "CLEAR!!":
            conv = get_conv_template("XuanYuan-Chat")
            print("clear...")
            continue

        conv.append_message(conv.roles[0], content.strip())
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if is_vllm_avaiable:
            outputs = infer_vllm(llm, sampling_params, prompt)
        else:
            outputs = infer(model, tokenizer, prompt)
        print(f"输出: {outputs}")
        conv.update_last_message(outputs)
