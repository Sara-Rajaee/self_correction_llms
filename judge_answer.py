import os
import json
import argparse
import numpy as np

from utils.data import load_generated_data
from utils.parser import extract_pred_and_parse
from utils.eval import per_sample_verification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--split", default="test", type=str)
    args = parser.parse_args()
    return args


def prepare_data(data_path, data_name, args):
    examples = load_generated_data(data_path)

    # Get out_file name
    model_name = args.model_name_or_path.split('/')[-1]
    out_file_prefix = f"{args.split}_{model_name}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    
    judge_file = f"{output_dir}/{data_name}/judge/{out_file_prefix}_judge.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/{data_name}/judge", exist_ok=True)
    return examples, judge_file


def setup(args):
    # Eval
    data_paths = args.data_path.split(",")
    data_list = args.data_names.split(",")
    assert len(data_list) == len(data_paths)

    for i, d in enumerate(data_list):
        main(d, data_paths[i], args)


def main(data_name, data_path, args):
    examples, judge_file = prepare_data(data_path, data_name, args)
    print("Data prepration done!")
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for example in examples:
        sample = {
            "idx": example["idx"],
            "question": example["question"],
            "gt": example['gt'],
            "prompt": example['prompt'],
            "first_reasoning": example['first_reasoning'],
            "think_sum": example['think_sum']
        }
        samples.append(sample)

    updated_samples = []
    for sample in samples:
        scores_per_prompt = []
        preds_per_prompt = []
        avg_score_per_first_reasoning = []
        for j in range(len(sample['first_reasoning'])):
            preds_per_first_reasoning = [extract_pred_and_parse(think_sum, data_name) for think_sum in sample['think_sum'][j]]
            scores_per_first_reasoning = per_sample_verification(preds_per_first_reasoning, sample['gt'])
            # Convert to string for saving
            preds_per_first_reasoning = [str(pred) for pred in preds_per_first_reasoning]
            
            scores_per_prompt.append(scores_per_first_reasoning)
            preds_per_prompt.append(preds_per_first_reasoning)
            avg_score_per_first_reasoning.append(np.mean(scores_per_first_reasoning))

        sample.update({
            "score": scores_per_prompt,
            "pred": preds_per_prompt,
            "avg_score_per_first_reasoning": avg_score_per_first_reasoning
        })
        updated_samples.append(sample)

    print(f"Save to {judge_file}")
    json.dump(updated_samples, open(judge_file, "w",), indent=2)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    setup(args)