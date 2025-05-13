import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import transformers
from vllm import LLM, SamplingParams

from utils.data import load_generated_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--min_p", default=0., type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--max_model_len', type=int, default=40000)
    parser.add_argument("--n_sampling", default=1, type=int, help="I.e. n")
    parser.add_argument("--first_reasoning", action="store_true", default=False)
    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


def prepare_data(args):
    examples = load_generated_data(args.data_path)

    # Get output file name
    model_name = args.model_name_or_path.split('/')[-1]
    out_file_prefix = args.data_path.split("/")[-1][:-len(".json")]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    
    prediction_file = f"{output_dir}/{out_file_prefix}_{model_name}_prediction.json"
    os.makedirs(f"{output_dir}", exist_ok=True)
    return examples, prediction_file


def setup(args):
    # Load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=0.96,
        enforce_eager=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Infer
    main(llm, tokenizer, args)


def build_conversation(tokenizer, question, first_reasoning=None):
    system_prompt = "Please reason step by step, and put your final answer within \boxed{}."
    user_prompt = f"Question: {question}"
    #if first_reasoning is not None:
    #    system_prompt = "Please solve the following math problem step by step. Continue exactly from where you left off, treating the previous reasoning as your own. Build your continued thinking based on it. Output only your full thinking process, and then summarize the final answer in \boxed{}."
    #    user_prompt = f"Question: {question}\n\nReasoning:{first_reasoning}"
    #else:
    #    user_prompt = f"Question: {question}"

    messages = [
        {
            "role": "system",
            "content": system_prompt,  
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if first_reasoning is not None:
        prompt = prompt + "<think>\n" + first_reasoning
    return prompt


def main(llm, tokenizer, args):
    examples, prediction_file = prepare_data(args)

    samples = []
    for _, example in tqdm(enumerate(examples), total=len(examples)):
        sample = {
            "id": example["id"],
            "idx": example["idx"],
            "question": example["question"],
            "gt": example['gt'],
            "prompt": build_conversation(tokenizer, example["question"], first_reasoning=example['first_reasoning']) if args.first_reasoning else build_conversation(tokenizer,  example["question"]),
            'first_reasoning': example['first_reasoning']
        }
        samples.append(sample)

    prompts = [sample["prompt"] for sample in samples]

    # Start inference
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens_per_call,
        min_tokens=2,
        n=args.n_sampling,
        skip_special_tokens=False,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    generated_reasonings = [output.outputs[i].text for output in outputs for i in range(args.n_sampling)] # flatten
    assert len(generated_reasonings) == len(prompts) * args.n_sampling

    # Save
    for sample, gen in zip(samples, generated_reasonings):
        sample.update({"reasoning_after_first": gen})

    print(f"Save to {prediction_file}")
    json.dump(samples, open(prediction_file, "w",), indent=2)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)