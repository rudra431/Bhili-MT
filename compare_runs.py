#!/usr/bin/env python3
"""
Compare multiple eval runs side-by-side.

Usage:
  python compare_runs.py eval_results/baseline_greedy eval_results/baseline_beam eval_results/beam_postedit
"""

import argparse, json, os, sys


def load_report(run_dir):
    p = os.path.join(run_dir, "report.json")
    if not os.path.exists(p):
        print(f"WARN: no report.json in {run_dir}", file=sys.stderr)
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    reports = [(d, load_report(d)) for d in args.run_dirs]
    reports = [(d, r) for d, r in reports if r is not None]
    if not reports:
        print("No valid reports found.", file=sys.stderr)
        return

    directions = []
    for _, r in reports:
        for res in r["results"]:
            if res["direction"] not in directions:
                directions.append(res["direction"])

    lines = []
    lines.append("=" * 115)
    lines.append("Side-by-side comparison")
    lines.append("=" * 115)
    for dirn in directions:
        lines.append(f"\n--- {dirn} ---")
        lines.append(f"{'Run':<40}{'chrF++':>10}{'COMET':>10}{'Align':>10}{'TransRate':>12}{'TransH/T':>14}")
        lines.append("-" * 96)
        for run_dir, rpt in reports:
            res = next((r for r in rpt["results"] if r["direction"] == dirn), None)
            if not res:
                continue
            cfg = rpt["config"]
            label_base = os.path.basename(run_dir.rstrip("/"))
            tag = f"[{cfg['decoding']}{'+pe' if cfg['use_glossary_postedit'] else ''}]"
            label = f"{label_base} {tag}"[:39]
            comet = f"{res['comet']:.4f}" if res['comet'] is not None else "-"
            if res['comet'] is not None and not res['comet_reliable']:
                comet += "*"
            align = f"{res['glossary_alignment']:.4f}" if res['glossary_alignment'] is not None else "-"
            trans = f"{res['translation_rate']:.4f}" if res['translation_rate'] is not None else "-"
            lines.append(f"{label:<40}{res['chrf2pp']:>10.4f}{comet:>10}"
                         f"{align:>10}{trans:>12}{res['translation_hits_total']:>14}")
    lines.append("\n* COMET unreliable for mar->bhb (Bhili not in COMET training data)")
    lines.append("Align     = alignment with glossary's prescribed bhb form")
    lines.append("TransRate = translation rate, restricted to mar != bhb terms (most sensitive to leakage)")

    out = "\n".join(lines)
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()