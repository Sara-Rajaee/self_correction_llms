from typing import Any, Dict
from math_verify import parse, StringExtractionConfig, LatexExtractionConfig
import sympy
import importlib

def parse_question(example, data_name):
    question = ""
    if data_name in ["mmlu_stem", "gpqa"]:
        options = example["choices"]
        assert len(options) == 4
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"{label}. {str(option).strip()}\n"
        options = " ".join(options).strip()
        question = f"{example['question'].strip()}\n\n {options}"
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    return question.strip()


def parse_ground_truth(example: Dict[str, Any], data_name):
    if data_name in ["math", "math500", "math500_level1", "math500_level2", 
                     "math500_level3", "math500_level4", "math500_level5",
                     "aime24", "aime25", "aimo2", "hmmt"]:
        answer = "$" + example["answer"] + "$"
    elif data_name in ["mmlu_stem", "gpqa"]:
        abcd = "ABCD"
        answer = "$" + abcd[example["answer"]] + "$"
    return parse(answer)


def extract_pred_and_parse(completion, data_name):
    if data_name in ["gpqa"]:
        pred = parse(
            completion,
            extraction_config=[StringExtractionConfig(lowercase=False)],
        )
    elif "boxed" in completion:
        pred = parse(
            completion, 
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, 
                    try_extract_without_anchor=True,
                ),
            ]
        )
    else:
        pred = []
    return pred


def serialize_list_of_lists(data):
    serialized = []
    for sublist in data:
        serialized.append([
            {
                "type": "native",
                "value": item
            } if isinstance(item, (int, float, str, bool, type(None))) else
            {
                "type": f"{type(item).__module__}.{type(item).__name__}",
                "value": str(item)
            }
            for item in sublist
        ])
    return {"data": serialized}

def deserialize_list_of_lists(serialized_dict):
    deserialized = []
    for sublist in serialized_dict["data"]:
        deserialized.append([
            item["value"] if item["type"] == "native" else
            getattr(importlib.import_module(item["type"].rsplit(".", 1)[0]), item["type"].split(".")[-1])(item["value"])
            for item in sublist
        ])
    return deserialized