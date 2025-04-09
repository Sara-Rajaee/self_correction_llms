import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams

from utils.data import load_data, construct_prompt
from utils.parser import parse_question, parse_ground_truth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--min_p", default=0., type=float)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--max_model_len', type=int, default=64000)
    parser.add_argument("--n_sampling", default=1, type=int, help="I.e. n")
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


def prepare_data(data_name, args):
    if "math500_level" in data_name:
        level = int(data_name.strip()[-1])
        examples = load_data("math500", args.split, args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(data_name, args.split, args.data_dir)

    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    # Select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # Get output file name
    model_name = args.model_name_or_path.split('/')[-1]
    out_file_prefix = f"{args.split}_{model_name}_seed{args.seed}_t{args.temperature}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    
    generated_dataset_file = f"{output_dir}/{data_name}/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_first_reasoning.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, generated_dataset_file


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
    )

    # Infer
    data_list = args.data_names.split(",")
    for data_name in data_list:
        main(llm, data_name, args)


def main(llm, data_name, args):
    examples, generated_dataset_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        idx = example["idx"]

        # Parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, args)

        if i == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": str(gt[0]),
            "prompt": full_prompt,
        }
        samples.append(sample)

    # Start generation
    stop_token = ["Alternatively,"]
    prompts = [sample["prompt"] for sample in samples] # for _ in range(args.n_sampling)]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens_per_call,
        min_tokens=2,
        n=args.n_sampling,
        skip_special_tokens=False,
        seed=args.seed,
        stop=stop_token,
    )
    outputs = llm.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    generated_reasonings = [output.outputs[i].text.rstrip() for output in outputs for i in range(args.n_sampling)]
    stop_reasons = [output.outputs[i].stop_reason for output in outputs for i in range(args.n_sampling)]
    assert len(generated_reasonings) == len(prompts) * args.n_sampling

    # Prepare output
    updated_samples = []
    for i, sample in enumerate(samples):
        sample_generated_reasonings = generated_reasonings[i * args.n_sampling : (i + 1) * args.n_sampling]
        sample_stop_reasons = stop_reasons[i * args.n_sampling : (i + 1) * args.n_sampling]
        selected_sample_generated_reasonings = []
        for j in range(len(sample_generated_reasonings)):
            if sample_stop_reasons[j] == stop_token[0]:
                selected_sample_generated_reasonings.append(sample_generated_reasonings[j])

        sample.update({"first_reasoning": selected_sample_generated_reasonings})
        updated_samples.append(sample)

    # Save
    print(f"Save to {generated_dataset_file}")
    json.dump(updated_samples, open(generated_dataset_file, "w",), indent=2)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)