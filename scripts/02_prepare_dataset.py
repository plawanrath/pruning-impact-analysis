"""Load BBQ dataset, filter to ambiguous condition, export as JSONL."""
import json
import os
import yaml
from datasets import load_dataset


UNKNOWN_MARKERS = [
    "unknown", "can't be determined", "not answerable",
    "not enough info", "undetermined", "cannot be determined",
    "not known", "not enough information",
]


def _extract_answer_info(example: dict) -> dict:
    answer_choices = [example["ans0"], example["ans1"], example["ans2"]]

    unknown_idx = None
    for i, ans in enumerate(answer_choices):
        if any(marker in ans.lower() for marker in UNKNOWN_MARKERS):
            unknown_idx = i
            break
    if unknown_idx is None:
        unknown_idx = example.get("answer_label", example.get("label", 2))

    non_unknown = [i for i in range(3) if i != unknown_idx]
    target_label = example.get("target_label")
    target_loc = example.get("target_loc", "")

    if target_label is not None:
        stereotype_idx = int(target_label)
    elif target_loc == "ans0":
        stereotype_idx = 0
    elif target_loc == "ans1":
        stereotype_idx = 1
    elif target_loc == "ans2":
        stereotype_idx = 2
    else:
        stereotype_idx = non_unknown[0]

    anti_stereotype_idx = [i for i in non_unknown if i != stereotype_idx]
    anti_stereotype_idx = anti_stereotype_idx[0] if anti_stereotype_idx else non_unknown[-1]

    return {
        "answer_choices": answer_choices,
        "stereotype_target_index": stereotype_idx,
        "anti_stereotype_target_index": anti_stereotype_idx,
        "unknown_index": unknown_idx,
    }


def prepare_dataset(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    ds_cfg = cfg["dataset"]

    output_path = ds_cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading {ds_cfg['name']}...")
    ds = load_dataset(ds_cfg["name"])

    total = 0
    kept = 0
    category_counts = {}

    with open(output_path, "w") as f:
        for split in ds:
            for example in ds[split]:
                total += 1
                # Filter to ambiguous condition
                context_condition = example.get("context_condition", "")
                if context_condition != ds_cfg["condition"]:
                    continue

                # Filter to target bias categories
                category = example.get("category", "")
                if category not in ds_cfg["bias_categories"]:
                    continue

                answer_info = _extract_answer_info(example)

                record = {
                    "item_id": example.get("example_id", kept),
                    "category": category,
                    "context": example["context"],
                    "question": example["question"],
                    "answer_choices": answer_info["answer_choices"],
                    "stereotype_target_index": answer_info["stereotype_target_index"],
                    "anti_stereotype_target_index": answer_info["anti_stereotype_target_index"],
                    "unknown_index": answer_info["unknown_index"],
                }
                f.write(json.dumps(record) + "\n")
                kept += 1
                category_counts[category] = category_counts.get(category, 0) + 1

    print(f"Total examples scanned: {total}")
    print(f"Kept (ambiguous, target categories): {kept}")
    print(f"Per-category counts:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    prepare_dataset()
