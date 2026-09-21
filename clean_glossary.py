#!/usr/bin/env python3
"""
Clean the Mahavistaar glossary TSV into a JSON with two lookup dicts.

Input:  Mahavistaar_Glossary_Main__1_.tsv
        (columns: dp_id, english, marathi, devhali_bhili, src_datapoint_id)
Output: glossary_clean.json
        (keys: mar2bhb, bhb2mar, stats)

Cleaning rules:
  - Drop empty entries
  - Drop Latin-script Bhili entries (~57 untranscribed romanizations)
  - For multi-variant entries ("X / Y / Z"), take first variant
  - KEEP mar == bhb entries — per field team, these are modern terms with
    no traditional Bhili equivalent, so Bhili speakers genuinely use the
    Marathi form
"""

import argparse, csv, json, re, sys
from collections import Counter


def normalize_variant(s: str) -> str:
    """Take first variant if /-separated, strip whitespace."""
    s = s.strip()
    if "/" in s:
        s = s.split("/")[0].strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) > 1 and all(len(p) > 1 for p in parts):
            s = parts[0]
    return s.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    mar2bhb, bhb2mar = {}, {}
    stats = Counter()

    with open(args.input, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stats["total"] += 1
            mar_raw = (row.get("marathi") or "").strip()
            bhb_raw = (row.get("devhali_bhili") or "").strip()

            if not mar_raw or not bhb_raw:
                stats["drop_empty"] += 1
                continue

            mar = normalize_variant(mar_raw)
            bhb = normalize_variant(bhb_raw)

            if not mar or not bhb:
                stats["drop_empty_after_normalize"] += 1
                continue

            if re.search(r"[a-zA-Z]", bhb):
                stats["drop_latin_bhili"] += 1
                continue

            if len(mar) < 2 or len(bhb) < 2:
                stats["drop_too_short"] += 1
                continue

            if re.sub(r"\s+", "", mar.lower()) == re.sub(r"\s+", "", bhb.lower()):
                stats["kept_mar_equals_bhb"] += 1

            if mar in mar2bhb:
                if len(bhb) < len(mar2bhb[mar]):
                    mar2bhb[mar] = bhb
                stats["collision_mar"] += 1
            else:
                mar2bhb[mar] = bhb
                stats["kept"] += 1

            if bhb in bhb2mar:
                if len(mar) < len(bhb2mar[bhb]):
                    bhb2mar[bhb] = mar
                stats["collision_bhb"] += 1
            else:
                bhb2mar[bhb] = mar

    out = {"mar2bhb": mar2bhb, "bhb2mar": bhb2mar, "stats": dict(stats)}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Glossary cleaning summary:", file=sys.stderr)
    print(f"  total rows read:           {stats['total']:>5}", file=sys.stderr)
    print(f"  dropped (empty):           {stats['drop_empty']:>5}", file=sys.stderr)
    print(f"  dropped (latin bhili):     {stats['drop_latin_bhili']:>5}", file=sys.stderr)
    print(f"  dropped (too short):       {stats['drop_too_short']:>5}", file=sys.stderr)
    print(f"  kept where mar==bhb:       {stats['kept_mar_equals_bhb']:>5}  (modern terms, per field team)", file=sys.stderr)
    print(f"  mar collisions (deduped):  {stats['collision_mar']:>5}", file=sys.stderr)
    print(f"  bhb collisions (deduped):  {stats['collision_bhb']:>5}", file=sys.stderr)
    print(f"  unique mar->bhb kept:      {len(mar2bhb):>5}", file=sys.stderr)
    print(f"  unique bhb->mar kept:      {len(bhb2mar):>5}", file=sys.stderr)
    print(f"\nWrote: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()