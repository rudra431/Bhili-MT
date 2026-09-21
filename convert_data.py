"""
Convert a TSV file with columns:
    dp_id, id, marathi, validated_translation
into a JSONL file with records of the form:
    {"id": ..., "category": ..., "eng": ..., "hin": ..., "mar": ..., "bhb": ...}
 
Usage:
    python tsv_to_jsonl.py input.tsv output.jsonl [--category advisory_pop]
"""
import argparse
import csv
import json
import sys
 
 
def convert(input_path, output_path, category=""):
    with open(input_path, "r", encoding="utf-8", newline="") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
 
        required = {"id", "marathi", "validated_translation"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"Error: input TSV is missing expected column(s): {', '.join(sorted(missing))}")
 
        count = 0
        for row in reader:
            # Skip completely blank lines that sometimes trail a TSV export
            if not any((row.get(k) or "").strip() for k in row):
                continue
 
            record = {
                "id": (row.get("id") or "").strip(),
                "category": category,
                "eng": "",
                "hin": "",
                "mar": (row.get("marathi") or "").strip(),
                "bhb": (row.get("validated_translation") or "").strip(),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
 
    print(f"Wrote {count} record(s) to {output_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TSV (dp_id, id, marathi, validated_translation) to JSONL.")
    parser.add_argument("input", help="Path to the input .tsv file")
    parser.add_argument("output", help="Path to the output .jsonl file")
    parser.add_argument("--category", default="", help="Value to fill in the 'category' field for every record (default: empty string)")
    args = parser.parse_args()
 
    convert(args.input, args.output, args.category)
 