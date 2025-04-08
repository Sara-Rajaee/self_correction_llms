import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams

from utils.data import load_generated_data
from utils.parser import extract_pred_and_parse
from utils.eval import per_sample_verification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--dataset_dir", default="", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
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
    parser.add_argument("--evaluation_mode", action='store_true', default=False)
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


def prepare_data(data_path, data_name, args):
    examples = load_generated_data(data_path)
    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # Get output file name
    model_name = args.model_name_or_path.split('/')[-1]
    out_file_prefix = f"{args.split}_{data_name}_{model_name}_seed{args.seed}_t{args.temperature}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    generated_dataset_file = f"{output_dir}/{data_name}/predictions/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_dataset_predictions.json"
    self_correction_file = f"{output_dir}/{data_name}/self_correction/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_self_correction.json"
    self_correction_performance = f"{output_dir}/{data_name}/self_correction/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_self_correction_performance.json"

    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/{data_name}/predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/{data_name}/self_correction", exist_ok=True)
    return examples, generated_dataset_file, self_correction_file, self_correction_performance


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
    tokenizer = llm.get_tokenizer()

    # Infer & eval
    data_paths = args.dataset_dir.split(",")
    data_list = args.data_names.split(",")
    assert len(data_list) == len(data_paths)

    for i, data_name in enumerate(data_list):
        main(llm, tokenizer, data_name, data_paths[i], args)


def main(llm, tokenizer, data_name, data_path, args):
    examples, generated_dataset_file, self_correction_file, self_correction_performance = prepare_data(data_path, data_name, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        sample = {
            "question": example["question"],
            "gt": example['gt'],
            "prompt": example['prompt'],
            'first_reasonings': example['first_reasonings']
        }
        if args.evaluation_mode:
            sample.update({"final_score_per_first_reasoning": example['final_score_per_first_reasoning']})

        samples.append(sample)

    prompts = []
    new_gt =[]
    if args.evaluation_mode:
        for i, sample in enumerate(samples):
            tmp_indices = []
            for j in range(len(sample['first_reasonings'])):
                if sample['final_score_per_first_reasoning'][j] == 0:
                    prompts.append(sample['prompt'] + sample['first_reasonings'][j] + "/n/nAlternatively,")
                    new_gt.append(sample['gt'])
                else:
                    tmp_indices.append(j)
            for j in sorted(tmp_indices, reverse=True):
                sample['first_reasonings'].pop(j)
                sample['final_score_per_first_reasoning'].pop(j)
    else:
        for i, sample in enumerate(samples):
            for j in range(len(sample['first_reasonings'])):
                for _ in range(args.n_sampling):
                    prompts.append(sample['prompt'] + sample['first_reasonings'][j] + "\n</think>\n\n")

    # Start inference
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens_per_call,
        min_tokens=2,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,

    )
    outputs = llm.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    generated_reasonings = [output.outputs[0].text for output in outputs]
    assert len(generated_reasonings) == len(prompts)

    
    if args.evaluation_mode:

        result= [extract_pred_and_parse(answer, data_name) for answer in generated_reasonings]
        performance = per_sample_verification(result, new_gt)
        accuracy = (sum(performance)/len(performance))

        
        start_idx = 0
        for i, sample in enumerate(samples):
            n_first_reasoning_sampling = len(sample['first_reasonings'])
            generated_answers = generated_reasonings[start_idx : start_idx + (n_first_reasoning_sampling)]
            start_idx = start_idx + n_first_reasoning_sampling
            sample.update({"answer": generated_answers})


        print(f"Save to {self_correction_file}")
        json.dump(samples, open(self_correction_file, "w",), indent=2)

        result_json = {
        "data_name": data_name,    
        "num_samples": len(performance),
        "acc": float(f"{accuracy:.4f}") * 100,
        }
        print(f"Save to {self_correction_performance}")
        json.dump(result_json, open(self_correction_performance, "w",), indent=2)

    else:
        answers = []
        start_idx = 0
        for i, sample in enumerate(samples):
            n_first_reasoning_sampling = len(sample['first_reasonings'])
            sample_answers = generated_reasonings[start_idx : start_idx + (n_first_reasoning_sampling * args.n_sampling)]
            assert len(sample_answers) == n_first_reasoning_sampling * args.n_sampling
            answers.append([sample_answers[i * args.n_sampling:(i + 1) * args.n_sampling] for i in range(n_first_reasoning_sampling)])
            start_idx = start_idx + (n_first_reasoning_sampling * args.n_sampling)

        updated_samples = []
        for i, sample in enumerate(samples):
            sample.update({"answer": answers[i]})
            updated_samples.append(sample)

        print(f"Save to {generated_dataset_file}")
        json.dump(updated_samples, open(generated_dataset_file, "w",), indent=2)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)