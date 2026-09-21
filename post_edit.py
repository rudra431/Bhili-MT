"""
Glossary post-edit for Bhili NMT.

Applies the field-team-curated glossary to translation outputs:
for each glossary source-term in the input, if the model copied the source-language
form through to the output instead of using the target-language form, substitute it.

Only fires when source and target glossary forms genuinely differ (i.e., not for
'modern term' entries where mar == bhb).

Usage:
    from postedit import load_glossary, post_edit

    g = load_glossary("glossary_clean.json")
    # for mar->bhb direction:
    edited, edits_applied = post_edit(
        source_text, model_output,
        g["mar2bhb_sorted_keys"], g["mar2bhb"]
    )
    # for bhb->mar direction:
    edited, edits_applied = post_edit(
        source_text, model_output,
        g["bhb2mar_sorted_keys"], g["bhb2mar"]
    )

--------------------------------------------------------------------------------
2026-09-15 fix: right-side word-boundary check.

Previously `_contains_term` / `_replace_with_boundary` only checked the LEFT
boundary of a candidate match and were unconditionally lenient on the right
("lenient on right for inflection"). That let a short glossary key match
*inside* an unrelated compound noun -- e.g. the entry 'बोंड' -> 'कवठ' matched
the 'बोंड' inside 'बोंडअळी' (bollworm) and corrupted it into 'कवठअळी'
(nonsense), because 'अळी' ("worm") isn't an inflectional suffix of 'बोंड', it's
a separate content word glued on as a compound.

Fix: a match's right side is now only "OK" if it's a real boundary (end of
string / non-Devanagari char) OR the text immediately following the match
starts with a known Marathi inflectional suffix (case markers like ला/त/च्या/
तील/...). Anything else (i.e., another Devanagari word stem glued straight on)
blocks the match, same as the left side already did.

This does NOT fix glossary entries that are wrong even at a clean word
boundary (e.g. 'मावा' -> 'खवा', where 'मावा' is itself already the correct
Marathi word and the glossary entry is simply bad data). Those need to be
found and corrected in glossary.json itself -- see audit_glossary.py.
--------------------------------------------------------------------------------
"""

import json
import re


_DEV_LETTER_RE = re.compile(r"[ऀ-ॿ]")

# Common Marathi inflectional / case-marker continuations. Not exhaustive by
# design -- kept short and high-precision so it stays conservative: when in
# doubt, the safer failure mode is "leave the leaked source form alone"
# rather than "corrupt a compound word". Extend this list if real inflected
# leaks are found slipping through uncaught.
_SUFFIXES = sorted([
    # vowel-sign ("matra") + case-marker, for consonant-final stems
    # (बोंड + ाला -> बोंडाला, "to the boll")
    "ांच्या", "ांची", "ांचा", "ांचे", "ांना", "ातील", "ासाठी", "ामध्ये", "ाकडे",
    "ाहून", "ापर्यंत", "ाप्रमाणे", "ाला", "ाना", "ाने", "ाशी", "ावर", "ात", "ाचा",
    "ाची", "ाचे",
    # bare case-marker, for vowel-final stems
    "साठी", "मध्ये", "कडे", "हून", "पर्यंत", "प्रमाणे",
    "च्या", "ची", "चा", "चे", "ला", "ना", "ने", "शी", "तील", "ावर", "ात",
    "ां", "त", "ो", "ी", "े", "ा",
], key=len, reverse=True)


def _is_dev_letter(ch):
    return bool(_DEV_LETTER_RE.match(ch))


def _norm_compare(s):
    """Lowercase + whitespace-strip for identity comparison."""
    return re.sub(r"\s+", "", s.lower())


def _right_boundary_ok(text, end_pos):
    """True if `end_pos` (just past a candidate match) is a real word
    boundary, or the text continues with a recognized inflectional suffix
    rather than an unrelated compound continuation."""
    if end_pos >= len(text):
        return True
    if not _is_dev_letter(text[end_pos]):
        return True
    remainder = text[end_pos:]
    return any(remainder.startswith(suf) for suf in _SUFFIXES)


def load_glossary(path):
    """Load cleaned glossary and pre-sort keys by length descending for longest-match scan.

    Returns dict with keys:
        - mar2bhb, bhb2mar:                  lookup dicts (term -> translation)
        - mar2bhb_sorted_keys, ...:          keys sorted by length desc
        - stats:                              cleaning metadata
    """
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    g["mar2bhb_sorted_keys"] = sorted(g["mar2bhb"].keys(), key=len, reverse=True)
    g["bhb2mar_sorted_keys"] = sorted(g["bhb2mar"].keys(), key=len, reverse=True)
    return g


def find_source_terms(text, sorted_keys, lookup):
    """Greedy longest-match scan of `text` for glossary source-terms with
    word-boundary constraints (no match inside another Devanagari word).

    Returns list of (source_term, target_term) in document order, non-overlapping.
    """
    found = []
    pos = 0
    n = len(text)
    while pos < n:
        matched = False
        for key in sorted_keys:
            klen = len(key)
            if klen == 0 or pos + klen > n:
                continue
            if text[pos:pos + klen] == key:
                before_ok = (pos == 0) or not _is_dev_letter(text[pos - 1])
                after_ok = (pos + klen == n) or not _is_dev_letter(text[pos + klen])
                if before_ok and after_ok:
                    found.append((key, lookup[key]))
                    pos += klen
                    matched = True
                    break
        if not matched:
            pos += 1
    return found


def _contains_term(text, term):
    """Substring match with boundary checks on both sides: left is strict
    (no match inside another word), right allows a real boundary or a known
    inflectional suffix (see module docstring for why)."""
    idx = 0
    while True:
        pos = text.find(term, idx)
        if pos == -1:
            return False
        left_ok = (pos == 0) or not _is_dev_letter(text[pos - 1])
        if left_ok and _right_boundary_ok(text, pos + len(term)):
            return True
        idx = pos + 1


def _replace_with_boundary(text, old, new):
    """Replace `old` with `new` only at genuine word boundaries on both sides
    (left strict, right allows a known inflectional suffix)."""
    out = []
    i = 0
    L = len(old)
    while i < len(text):
        if text[i:i + L] == old:
            left_ok = (i == 0) or not _is_dev_letter(text[i - 1])
            if left_ok and _right_boundary_ok(text, i + L):
                out.append(new)
                i += L
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def post_edit(source, prediction, sorted_keys, lookup):
    """Substitute leaked source-form glossary terms with their target equivalents.

    A substitution fires when:
      1. A glossary source-term appears in `source`, AND
      2. The source-term (not the target-term) still appears in `prediction`, AND
      3. The glossary entry has source != target (no-op for "modern term" entries
         where the field team prescribes keeping the source form).

    Args:
        source:       the original input sentence
        prediction:   the model's translation output
        sorted_keys:  glossary keys sorted longest-first (from load_glossary)
        lookup:       glossary dict (from load_glossary)

    Returns:
        (edited_prediction, list_of_(source_term, target_term)_pairs_applied)
    """
    src_terms = find_source_terms(source, sorted_keys, lookup)
    edited = prediction
    edits = []
    for src_t, tgt_t in src_terms:
        if _norm_compare(src_t) == _norm_compare(tgt_t):
            continue  # modern term — no substitution
        if _contains_term(edited, src_t) and not _contains_term(edited, tgt_t):
            new = _replace_with_boundary(edited, src_t, tgt_t)
            if new != edited:
                edits.append((src_t, tgt_t))
                edited = new
    return edited, edits
