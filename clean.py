import argparse
import json
import re
import sys
import unicodedata
from collections import Counter

# ─── Fields ───
EN = "eng"
MR = "mar"
BH = "bhb"


# ═══════════════════════════════════════════════
# 1. TEXT NORMALIZATION
# ═══════════════════════════════════════════════

def normalize_unicode(text):
    """NFC normalization — critical for Devanagari consistency."""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text):
    """Collapse multiple spaces, strip, normalize line breaks."""
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_invisible_chars(text):
    """Remove zero-width chars, BOM, and other invisible Unicode."""
    invisible = [
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\ufeff",  # BOM
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u00ad",  # soft hyphen
        "\u2060",  # word joiner
    ]
    for ch in invisible:
        text = text.replace(ch, "")
    return text


def normalize_punctuation(text):
    """Normalize common punctuation variants without removing them."""
    # Fullwidth → ASCII
    text = text.replace("\uff0c", ",")  # fullwidth comma
    text = text.replace("\uff0e", ".")  # fullwidth period
    text = text.replace("\uff1f", "?")  # fullwidth question mark
    text = text.replace("\uff01", "!")  # fullwidth exclamation
    text = text.replace("\uff1a", ":")  # fullwidth colon
    text = text.replace("\uff1b", ";")  # fullwidth semicolon
    # Curly quotes → straight
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Multiple periods → single
    text = re.sub(r"\.{4,}", "...", text)
    return text


def clean_stray_quotes(text):
    """Remove stray leading/trailing quotes that aren't part of the text."""
    text = re.sub(r'^["\u201c\u201d]+\s*', "", text)
    text = re.sub(r'\s*["\u201c\u201d]+$', "", text)
    return text


def normalize_text(text):
    """Full normalization pipeline for a single text field."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = remove_invisible_chars(text)
    text = normalize_punctuation(text)
    text = clean_stray_quotes(text)
    text = clean_whitespace(text)
    return text


# ═══════════════════════════════════════════════
# 2. SCRIPT DETECTION
# ═══════════════════════════════════════════════

def get_script_ratio(text):
    """Return ratio of Devanagari vs Latin characters."""
    if not text:
        return 0.0, 0.0
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    latin = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    total = len(text.replace(" ", ""))
    if total == 0:
        return 0.0, 0.0
    return devanagari / total, latin / total


def is_devanagari_dominant(text, threshold=0.3):
    """Check if text is primarily Devanagari."""
    dev_ratio, _ = get_script_ratio(text)
    return dev_ratio >= threshold


def is_latin_dominant(text, threshold=0.3):
    """Check if text is primarily Latin script."""
    _, lat_ratio = get_script_ratio(text)
    return lat_ratio >= threshold


# ═══════════════════════════════════════════════
# 3. FILTERING CHECKS
# ═══════════════════════════════════════════════

def check_record(record, min_chars=20, min_words=4, max_len_ratio=3.5, max_digit_ratio=0.5):
    """
    Run all quality checks on a record.
    Returns (is_valid, list_of_reasons).
    """
    reasons = []

    en = record.get(EN, "").strip()
    mr = record.get(MR, "").strip()
    bh = record.get(BH, "").strip()

    # ── Empty / near-empty ──
    if not mr:
        reasons.append("empty_marathi")
    if not bh:
        reasons.append("empty_bhili")
    if not en:
        reasons.append("empty_english")

    # If both source and target are empty, no point checking further
    if not mr or not bh:
        return len(reasons) == 0, reasons

    # ── Too short ──
    if len(mr) < min_chars:
        reasons.append(f"marathi_too_short ({len(mr)} chars)")
    if len(bh) < min_chars:
        reasons.append(f"bhili_too_short ({len(bh)} chars)")
    if en and len(en) < 5:
        reasons.append(f"english_too_short ({len(en)} chars)")

    mr_words = len(mr.split())
    bh_words = len(bh.split())
    if mr_words < min_words:
        reasons.append(f"marathi_few_words ({mr_words})")
    if bh_words < min_words:
        reasons.append(f"bhili_few_words ({bh_words})")

    # ── Length ratio (misalignment) ──
    if mr_words > 0 and bh_words > 0:
        ratio = max(mr_words / bh_words, bh_words / mr_words)
        if ratio > max_len_ratio:
            reasons.append(f"length_ratio ({ratio:.1f}x)")

    # ── Untranslated (source == target) ──
    mr_norm = re.sub(r"\s+", "", mr.lower())
    bh_norm = re.sub(r"\s+", "", bh.lower())
    if mr_norm == bh_norm:
        reasons.append("untranslated_copy")

    # Very high similarity (>90% character overlap)
    if mr_norm and bh_norm:
        common = sum(1 for a, b in zip(mr_norm, bh_norm) if a == b)
        similarity = common / max(len(mr_norm), len(bh_norm))
        if similarity > 0.90 and len(mr_norm) > 10:
            reasons.append(f"near_duplicate ({similarity:.0%} similar)")

    # ── Script mismatch ──
    if mr and not is_devanagari_dominant(mr):
        reasons.append("marathi_not_devanagari")
    if bh and not is_devanagari_dominant(bh):
        reasons.append("bhili_not_devanagari")
    if en and is_devanagari_dominant(en):
        reasons.append("english_has_devanagari")

    # ── Excessive digits ──
    for field_name, text in [("marathi", mr), ("bhili", bh)]:
        if text:
            digits = sum(1 for c in text if c.isdigit())
            alpha = sum(1 for c in text if c.isalpha())
            if alpha > 0 and digits / (digits + alpha) > max_digit_ratio:
                reasons.append(f"{field_name}_excessive_digits")

    # ── Corrupted Unicode ──
    for field_name, text in [("marathi", mr), ("bhili", bh), ("english", en)]:
        if text:
            if "\ufffd" in text:
                reasons.append(f"{field_name}_corrupted_unicode")
            # Check for unusual control characters
            if any(unicodedata.category(c).startswith("C") and c not in "\n\r\t" for c in text):
                reasons.append(f"{field_name}_control_chars")

    # ── Repetitive content ──
    for field_name, text in [("marathi", mr), ("bhili", bh)]:
        if text:
            words = text.split()
            if len(words) > 5:
                # Check if any word repeats more than 40% of the time
                word_counts = Counter(words)
                most_common_count = word_counts.most_common(1)[0][1]
                if most_common_count / len(words) > 0.4:
                    reasons.append(f"{field_name}_repetitive")

    return len(reasons) == 0, reasons


# ═══════════════════════════════════════════════
# 4. DEDUPLICATION
# ═══════════════════════════════════════════════

def dedup_key(record):
    """Generate dedup key from normalized Marathi + Bhili."""
    mr = re.sub(r"\s+", "", record.get(MR, "").lower())
    bh = re.sub(r"\s+", "", record.get(BH, "").lower())
    return f"{mr}|||{bh}"


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Clean and filter NMT training data")
    p.add_argument("--input", nargs="+", required=True, help="Input JSONL files")
    p.add_argument("--output", required=True, help="Output clean JSONL")
    p.add_argument("--rejected", default="rejected.jsonl", help="Rejected samples log")
    p.add_argument("--min_chars", type=int, default=20, help="Min characters per field (default: 20)")
    p.add_argument("--min_words", type=int, default=4, help="Min words per field (default: 4)")
    p.add_argument("--max_ratio", type=float, default=3.5, help="Max source/target length ratio (default: 3.5)")
    p.add_argument("--max_digit_ratio", type=float, default=0.5, help="Max digit ratio (default: 0.5)")
    p.add_argument("--stats", action="store_true", help="Print detailed stats")
    args = p.parse_args()

    # ── Load all inputs ──
    all_records = []
    for filepath in args.input:
        with open(filepath, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
                    count += 1
        print(f"Loaded {count} from {filepath}")
    print(f"Total loaded: {len(all_records)}")

    # ── Normalize ──
    print("Normalizing...")
    for rec in all_records:
        rec[EN] = normalize_text(rec.get(EN, ""))
        rec[MR] = normalize_text(rec.get(MR, ""))
        rec[BH] = normalize_text(rec.get(BH, ""))

    # ── Filter ──
    print("Filtering...")
    clean = []
    rejected = []
    reason_counts = Counter()

    for rec in all_records:
        is_valid, reasons = check_record(
            rec,
            min_chars=args.min_chars,
            min_words=args.min_words,
            max_len_ratio=args.max_ratio,
            max_digit_ratio=args.max_digit_ratio,
        )
        if is_valid:
            clean.append(rec)
        else:
            rec["_rejection_reasons"] = reasons
            rejected.append(rec)
            for r in reasons:
                # Normalize reason for counting (strip parenthetical details)
                base_reason = re.sub(r"\s*\(.*\)", "", r)
                reason_counts[base_reason] += 1

    print(f"After filtering: {len(clean)} clean, {len(rejected)} rejected")

    # ── Dedup ──
    print("Deduplicating...")
    seen = set()
    deduped = []
    dup_count = 0
    for rec in clean:
        key = dedup_key(rec)
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
        else:
            dup_count += 1
            rec["_rejection_reasons"] = ["duplicate"]
            rejected.append(rec)
            reason_counts["duplicate"] += 1

    print(f"After dedup: {len(deduped)} clean, {dup_count} duplicates removed")

    # ── Write outputs ──
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(args.rejected, "w", encoding="utf-8") as f:
        for rec in rejected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Stats ──
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Input:    {len(all_records)}")
    print(f"Clean:    {len(deduped)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Rate:     {len(deduped)/len(all_records)*100:.1f}% kept")
    print(f"\nOutputs:")
    print(f"  Clean:    {args.output}")
    print(f"  Rejected: {args.rejected}")

    if reason_counts:
        print(f"\nRejection reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

    if args.stats:
        # Length distribution of clean data
        mr_lens = [len(r[MR].split()) for r in deduped]
        bh_lens = [len(r[BH].split()) for r in deduped]
        print(f"\nClean data stats:")
        print(f"  Marathi words: min={min(mr_lens)}, max={max(mr_lens)}, avg={sum(mr_lens)/len(mr_lens):.0f}")
        print(f"  Bhili words:   min={min(bh_lens)}, max={max(bh_lens)}, avg={sum(bh_lens)/len(bh_lens):.0f}")

        # Category distribution
        cats = Counter(r.get("category", "unknown") for r in deduped)
        print(f"\nCategory distribution:")
        for cat, count in cats.most_common():
            print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()