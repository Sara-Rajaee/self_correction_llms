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
from utils.eval import per_pred_verification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_paths", default="", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--min_p", default=0., type=float)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--max_model_len', type=int, default=64000)
    parser.add_argument("--n_sampling", default=1, type=int, help="I.e. n")
    parser.add_argument("--eval_mode", action='store_true', default=False, 
        help=("When False, force the model to do prediction with the first reasoning. "
              "When True, evaluate the self-correction ability of a LLM")
    )
    parser.add_argument("--score_threshold", type=float, default=0)
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


def prepare_data(data_path, args):
    examples = load_generated_data(data_path)

    # Get output file name
    model_name = args.model_name_or_path.split('/')[-1]
    if not args.eval_mode:
        out_file_prefix = data_path[:-len(".json")]
    else:
        out_file_prefix = data_path[:-len("_judge.json")]
    prediction_file = f"{out_file_prefix}_{model_name}_prediction.json"
    self_correction_file = f"{out_file_prefix}_{model_name}_thres{args.score_threshold}_self_correction.json"
    self_correction_performance_file = f"{out_file_prefix}_{model_name}_thres{args.score_threshold}_self_correction_performance.json"
    return examples, prediction_file, self_correction_file, self_correction_performance_file


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
    data_paths = args.data_paths.split(",")
    data_list = args.data_names.split(",")
    assert len(data_list) == len(data_paths)

    for i, data_name in enumerate(data_list):
        main(llm, data_name, data_paths[i], args)


def main(llm, data_name, data_path, args):
    examples, prediction_file, self_correction_file, self_correction_performance_file = prepare_data(data_path, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for _, example in tqdm(enumerate(examples), total=len(examples)):
        sample = {
            "idx": example["idx"],
            "question": example["question"],
            "gt": example['gt'],
            "prompt": example['prompt'],
            'first_reasoning': example['first_reasoning']
        }
        if args.eval_mode:
            sample.update({"avg_score_per_first_reasoning": example["avg_score_per_first_reasoning"]})
        samples.append(sample)

    prompts = []
    if args.eval_mode:
        new_samples = []
        for sample in samples:
            if 0 in sample["avg_score_per_first_reasoning"]:
                tmp_indices = []
                for j in range(len(sample['first_reasoning'])):
                    if sample["avg_score_per_first_reasoning"][j] <= args.score_threshold:  # Choose the first reasoning with 0 avg score
                        prompts.append(sample['prompt'] + sample['first_reasoning'][j] + "\n\nAlternatively,")
                    else:
                        tmp_indices.append(j)
                # Delete the first reaosnings with avg score > 0
                for j in sorted(tmp_indices, reverse=True): 
                    sample['first_reasoning'].pop(j)
                    sample['avg_score_per_first_reasoning'].pop(j)
                new_samples.append(sample)
        samples = new_samples
        print(f"#Valid Samples: {len(samples)}, #First Reasonings: {len(prompts)}")
    else:
        for sample in samples:
            for j in range(len(sample['first_reasoning'])):
                prompts.append(sample['prompt'] + sample['first_reasoning'][j] + "\n</think>\n\n")

    # Start inference
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
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
    # Force the LLM to summatize the reasoning and do prediciton
    if not args.eval_mode:
        start_idx = 0
        for sample in samples:
            n_first_reasoning_per_prompt = len(sample['first_reasoning'])
            think_sum_per_prompt = generated_reasonings[start_idx : start_idx + (n_first_reasoning_per_prompt * args.n_sampling)]
            assert len(think_sum_per_prompt) == n_first_reasoning_per_prompt * args.n_sampling

            think_sum_per_prompt = [think_sum_per_prompt[i * args.n_sampling : (i + 1) * args.n_sampling] for i in range(n_first_reasoning_per_prompt)]
            sample.update({"think_sum": think_sum_per_prompt})  # Everything after </think>
            start_idx = start_idx + (n_first_reasoning_per_prompt * args.n_sampling)

        print(f"Save to {prediction_file}")
        json.dump(samples, open(prediction_file, "w",), indent=2)

    # Evaluate LLM's self-correction ability
    else:
        all_scores = []
        all_preds = [extract_pred_and_parse(reasoning, data_name) for reasoning in generated_reasonings]
        start_idx = 0
        for sample in samples:
            n_first_reasoning_per_prompt = len(sample['first_reasoning'])
            reasoning_after_first_per_prompt = generated_reasonings[start_idx : start_idx + (n_first_reasoning_per_prompt * args.n_sampling)]
            preds_per_prompt = all_preds[start_idx : start_idx + (n_first_reasoning_per_prompt * args.n_sampling)]
            scores_per_prompt = [per_pred_verification(pred, sample["gt"]) for pred in preds_per_prompt]
            preds_per_prompt = [str(pred) for pred in preds_per_prompt] # Convert to string
            all_scores.extend(scores_per_prompt)

            reasoning_after_first_per_prompt = [reasoning_after_first_per_prompt[j * args.n_sampling : (j + 1) * args.n_sampling] for j in range(n_first_reasoning_per_prompt)]
            preds_per_prompt = [preds_per_prompt[j * args.n_sampling : (j + 1) * args.n_sampling] for j in range(n_first_reasoning_per_prompt)]
            scores_per_prompt = [scores_per_prompt[j * args.n_sampling : (j + 1) * args.n_sampling] for j in range(n_first_reasoning_per_prompt)]
            start_idx = start_idx + (n_first_reasoning_per_prompt * args.n_sampling)

            sample.update({
                "reasoning_after_first": reasoning_after_first_per_prompt,
                "pred": preds_per_prompt,
                "score": scores_per_prompt,    
            })

        print(f"Save to {self_correction_file}")
        json.dump(samples, open(self_correction_file, "w",), indent=2)

        acc = np.mean(all_scores)
        print(f"False to Correct Accuracy: {acc}")
        result_json = {   
            "num_samples": len(samples),
            "num_first_reasoning": len(prompts),
            "num_pres": len(all_scores),
            "acc": float(f"{acc:.4f}") * 100,
        }
        print(f"Save to {self_correction_performance_file}")
        json.dump(result_json, open(self_correction_performance_file, "w",), indent=2)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)