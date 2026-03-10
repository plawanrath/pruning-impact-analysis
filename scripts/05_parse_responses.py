"""Re-parse raw responses with the latest parse_response() logic."""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.response_parser import parse_response


def reparse(raw_dir="results/raw", output_dir="results/reparsed", inplace=False):
    pattern = os.path.join(raw_dir, "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No JSONL files found in {raw_dir}")
        return

    if not inplace:
        os.makedirs(output_dir, exist_ok=True)

    total_records = 0
    changed = 0
    still_none = 0

    for filepath in files:
        basename = os.path.basename(filepath)
        records = []

        with open(filepath) as f:
            for line in f:
                rec = json.loads(line)
                old_answer = rec.get("parsed_answer")
                new_answer = parse_response(rec["raw_response"])
                if new_answer != old_answer:
                    changed += 1
                rec["parsed_answer"] = new_answer
                if new_answer is None:
                    still_none += 1
                total_records += 1
                records.append(rec)

        if inplace:
            out_path = filepath
        else:
            out_path = os.path.join(output_dir, basename)

        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    print(f"Files processed: {len(files)}")
    print(f"Total records:   {total_records}")
    print(f"Changed:         {changed}")
    print(f"Still None:      {still_none}")
    if not inplace:
        print(f"Output dir:      {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-parse BBQ raw responses")
    parser.add_argument("--raw-dir", default="results/raw", help="Directory with raw JSONL files")
    parser.add_argument("--output-dir", default="results/reparsed", help="Output directory (default: results/reparsed)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original files instead of writing to output-dir")
    args = parser.parse_args()
    reparse(raw_dir=args.raw_dir, output_dir=args.output_dir, inplace=args.inplace)
