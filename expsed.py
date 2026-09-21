#!/usr/bin/env python3
"""
Export post-edit examples to CSV for native-speaker review.

For each direction, writes one CSV with columns:
  category, source, reference, prediction_raw, prediction_final,
  post_edits (comma-separated "src->tgt"), reviewer_verdict, reviewer_notes

The reviewer_verdict and reviewer_notes columns are empty — meant to be
filled in by the Bhili native speaker. Suggested verdict values:
  postedit_better | raw_better | equal | both_wrong

Usage:
  # all post-edited rows
  python export_postedit_samples.py --run_dir eval_results/beam_postedit --out_dir review_csvs

  # random sample of 30 per direction
  python export_postedit_samples.py --run_dir eval_results/beam_postedit --out_dir review_csvs --sample 30

  # include rows where no post-edit happened too (for context — usually skip)
  python export_postedit_samples.py --run_dir eval_results/beam_postedit --out_dir review_csvs --include_unedited
"""

import argparse, csv, json, os, random, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Directory containing {direction}_predictions.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample", type=int, default=None,
                    help="Random sample size per direction (default: all edited rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_unedited", action="store_true",
                    help="Also include rows where post_edits is empty")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    for direction in ["mar2bhb", "bhb2mar"]:
        pred_path = os.path.join(args.run_dir, f"{direction}_predictions.jsonl")
        if not os.path.exists(pred_path):
            print(f"  skip {direction}: {pred_path} not found", file=sys.stderr)
            continue

        rows = []
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not args.include_unedited and not r.get("post_edits"):
                    continue
                # post_edits is a list of [src, tgt] pairs; format as readable string
                edits = r.get("post_edits", []) or []
                edits_str = "; ".join(f"{a} -> {b}" for a, b in edits)
                # detect if raw and final differ — sanity check
                changed = (r["prediction_raw"] != r["prediction_final"])
                rows.append({
                    "category": r.get("category", ""),
                    "source": r["source"],
                    "reference": r["reference"],
                    "prediction_raw": r["prediction_raw"],
                    "prediction_final": r["prediction_final"],
                    "changed_by_postedit": "yes" if changed else "no",
                    "post_edits": edits_str,
                    "num_edits": len(edits),
                    "reviewer_verdict": "",
                    "reviewer_notes": "",
                })

        if args.sample and len(rows) > args.sample:
            rows = random.sample(rows, args.sample)
            # sort by num_edits desc so reviewer sees most-edited first
            rows.sort(key=lambda r: -r["num_edits"])

        out_path = os.path.join(args.out_dir, f"{direction}_review.csv")
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            # utf-8-sig adds BOM so Excel renders Devanagari correctly on Windows
            writer = csv.DictWriter(f, fieldnames=[
                "category", "source", "reference",
                "prediction_raw", "prediction_final",
                "changed_by_postedit", "post_edits", "num_edits",
                "reviewer_verdict", "reviewer_notes",
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        n_edited = sum(1 for r in rows if r["changed_by_postedit"] == "yes")
        print(f"  {direction}: wrote {len(rows)} rows ({n_edited} actually changed by post-edit)"
              f" -> {out_path}", file=sys.stderr)

    print(f"\nDone. Suggested verdict values for the reviewer:", file=sys.stderr)
    print(f"  postedit_better | raw_better | equal | both_wrong", file=sys.stderr)


if __name__ == "__main__":
    main()