import os
import json
import argparse
import numpy as np

from utils.data import load_generated_data
from utils.parser import extract_pred_and_parse
from utils.eval import per_sample_verification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--data_paths", default="", type=str)
    args = parser.parse_args()
    return args


def prepare_data(data_path):
    examples = load_generated_data(data_path)

    # Get out_file name
    out_file_prefix = data_path[:-len(".json")]
    judge_file = f"{out_file_prefix}_judge.json"
    return examples, judge_file


def setup(args):
    # Eval
    data_paths = args.data_paths.split(",")
    data_list = args.data_names.split(",")
    assert len(data_list) == len(data_paths)

    for i, d in enumerate(data_list):
        main(d, data_paths[i], args)


def main(data_name, data_path, args):
    examples, judge_file = prepare_data(data_path)
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