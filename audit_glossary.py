#!/usr/bin/env python3
"""
Data-driven audit of glossary post-edit entries: does applying each glossary
substitution actually move predictions closer to the reference, or further
away?

Rationale: with ~5,700 bhb2mar entries (and a similar count for mar2bhb),
manually reviewing the glossary for bad entries (like 'मावा' -> 'खवा', which
corrupts an already-correct word) isn't practical. But you already have the
raw material to find these automatically: your eval runs saved
prediction_raw (before post-edit) and prediction_final (after post-edit)
alongside the reference, for every sample, across every checkpoint. Wherever
post_edits fired, we can score whether that specific substitution helped or
hurt, using sentence-level chrF++ against the reference as the yardstick.

This aggregates that across every *_predictions.jsonl file you point it at,
per (source_term, target_term) glossary entry, and ranks entries by net harm
so you know which ones to fix or remove first -- instead of reading the
glossary end to end.

Usage:
    pip install sacrebleu
    python audit_glossary.py "output/rudra-bhili-translate/generations/bhb2mar/*/bhb2mar_predictions.jsonl"
    python audit_glossary.py "output/rudra-bhili-translate/generations/mar2bhb/*/mar2bhb_predictions.jsonl"

    # both directions in one run, written to one report:
    python audit_glossary.py \
        "output/rudra-bhili-translate/generations/bhb2mar/*/bhb2mar_predictions.jsonl" \
        "output/rudra-bhili-translate/generations/mar2bhb/*/mar2bhb_predictions.jsonl" \
        --output glossary_audit.csv

Output columns:
    source_term, target_term, n_fired, n_helped, n_hurt, n_neutral,
    net_chrf_delta, avg_chrf_delta, example_file, example_idx

Sorted worst-first (most negative net_chrf_delta) so the harmful entries are
at the top. A large negative net_chrf_delta with many n_hurt and few n_helped
is a strong candidate to fix or delete from glossary.json; entries with a
solidly positive net_chrf_delta are doing their job.

Note: this only sees entries that actually fired somewhere in the files you
give it. An entry that never matched any eval sentence won't show up here --
it isn't wrong, it's just untested by this method.
"""

import argparse
import glob
import json
import sys
from collections import defaultdict

try:
    from sacrebleu.metrics import CHRF
except ImportError:
    print("ERROR: pip install sacrebleu", file=sys.stderr)
    sys.exit(1)

_chrf = CHRF(word_order=2)


def sentence_chrf(hyp, ref):
    if not hyp.strip():
        return 0.0
    return _chrf.sentence_score(hyp, [ref]).score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="glob pattern(s) for *_predictions.jsonl files")
    parser.add_argument("--output", default="glossary_audit.csv")
    args = parser.parse_args()

    files = []
    for pat in args.patterns:
        files.extend(sorted(glob.glob(pat, recursive=True)))
    if not files:
        print("No files matched the given pattern(s).", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(files)} file(s)...")

    # (source_term, target_term) -> stats
    stats = defaultdict(lambda: {
        "n_fired": 0, "n_helped": 0, "n_hurt": 0, "n_neutral": 0,
        "sum_delta": 0.0, "example_file": None, "example_idx": None,
        "example_src": None, "example_raw": None, "example_final": None,
    })

    n_rows_with_edits = 0
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                edits = ex.get("post_edits") or []
                if not edits:
                    continue
                n_rows_with_edits += 1

                ref = ex["reference"]
                raw = ex["prediction_raw"]
                final = ex["prediction_final"]

                chrf_raw = sentence_chrf(raw, ref)
                chrf_final = sentence_chrf(final, ref)
                delta = chrf_final - chrf_raw  # positive = post-edit helped this sentence

                # If multiple edits fired on the same sentence, we can't cleanly
                # attribute the delta to just one -- split it evenly across the
                # edits that fired here so multi-edit sentences don't over- or
                # under-count any single entry.
                share = delta / len(edits)

                for src_t, tgt_t in edits:
                    key = (src_t, tgt_t)
                    s = stats[key]
                    s["n_fired"] += 1
                    s["sum_delta"] += share
                    if share > 0.01:
                        s["n_helped"] += 1
                    elif share < -0.01:
                        s["n_hurt"] += 1
                    else:
                        s["n_neutral"] += 1
                    if s["example_file"] is None or share < 0:
                        # prefer keeping a harmful example visible
                        s["example_file"] = fpath
                        s["example_idx"] = idx
                        s["example_src"] = ex["source"]
                        s["example_raw"] = raw
                        s["example_final"] = final

    print(f"{n_rows_with_edits} rows had at least one post-edit; "
          f"{len(stats)} distinct glossary entries fired.")

    rows = []
    for (src_t, tgt_t), s in stats.items():
        rows.append({
            "source_term": src_t,
            "target_term": tgt_t,
            "n_fired": s["n_fired"],
            "n_helped": s["n_helped"],
            "n_hurt": s["n_hurt"],
            "n_neutral": s["n_neutral"],
            "net_chrf_delta": round(s["sum_delta"], 3),
            "avg_chrf_delta": round(s["sum_delta"] / s["n_fired"], 4),
            "example_file": s["example_file"],
            "example_idx": s["example_idx"],
            "example_source": s["example_src"],
            "example_raw": s["example_raw"],
            "example_final": s["example_final"],
        })

    rows.sort(key=lambda r: r["net_chrf_delta"])  # most harmful first

    import csv
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} glossary entries to {args.output}, worst first.\n")
    print("Top 10 most harmful entries (net_chrf_delta < 0 means post-edit hurt on net):")
    for r in rows[:10]:
        print(f"  {r['source_term']!r:20} -> {r['target_term']!r:20} "
              f"fired={r['n_fired']:>3} helped={r['n_helped']:>3} hurt={r['n_hurt']:>3} "
              f"net_delta={r['net_chrf_delta']:>8}")

    helpful = sorted(rows, key=lambda r: -r["net_chrf_delta"])
    print("\nTop 10 most helpful entries:")
    for r in helpful[:10]:
        print(f"  {r['source_term']!r:20} -> {r['target_term']!r:20} "
              f"fired={r['n_fired']:>3} helped={r['n_helped']:>3} hurt={r['n_hurt']:>3} "
              f"net_delta={r['net_chrf_delta']:>8}")


if __name__ == "__main__":
    main()
