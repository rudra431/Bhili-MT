#!/usr/bin/env python3
"""
Upload glossary.json and post_edit.py to the existing AI4Bharat HF repos.

Setup:
    pip install huggingface_hub
    huggingface-cli login        # one-time

Run:
    # dry run to preview
    python push_to_hf.py --dry_run

    # real upload
    python push_to_hf.py

    # only one repo (testing)
    python push_to_hf.py --repos ai4bharat/bhili-translate-mar-bhb
"""

import argparse
import os
import sys
from huggingface_hub import HfApi, upload_file


DEFAULT_REPOS = [
    "ai4bharat/bhili-translate-mar-bhb",
    "ai4bharat/bhili-translate-bhb-mar",
]

# (local_path, path_in_repo)
FILES = [
    ("glossary.json", "glossary.json"),
    ("post_edit.py", "post_edit.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    ap.add_argument("--local_dir", default=".",
                    help="Directory containing glossary.json and post_edit.py")
    ap.add_argument("--commit_message",
                    default="Add glossary + post-edit module for inference recipe")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="Skip interactive confirmation")
    args = ap.parse_args()

    api = HfApi()
    try:
        user = api.whoami()
        print(f"Authenticated as: {user.get('name', user)}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: not authenticated. Run `huggingface-cli login` first.", file=sys.stderr)
        print(f"  underlying error: {e}", file=sys.stderr)
        sys.exit(1)

    # verify files exist
    missing = []
    for local_name, _ in FILES:
        path = os.path.join(args.local_dir, local_name)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print(f"ERROR: missing files locally:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    # plan
    print(f"\nPlan:", file=sys.stderr)
    for repo in args.repos:
        print(f"  {repo}:", file=sys.stderr)
        for local_name, remote_name in FILES:
            sz = os.path.getsize(os.path.join(args.local_dir, local_name))
            print(f"    {local_name:<25} -> {remote_name}  ({sz:,} bytes)", file=sys.stderr)

    if args.dry_run:
        print(f"\n--dry_run: not uploading.", file=sys.stderr)
        return

    if not args.yes:
        confirm = input("\nProceed with upload? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.", file=sys.stderr)
            return

    # upload
    n_ok = n_fail = 0
    for repo in args.repos:
        print(f"\nUploading to {repo}...", file=sys.stderr)
        for local_name, remote_name in FILES:
            local_path = os.path.join(args.local_dir, local_name)
            try:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_name,
                    repo_id=repo,
                    repo_type="model",
                    commit_message=args.commit_message,
                )
                print(f"  ✓ {remote_name}", file=sys.stderr)
                n_ok += 1
            except Exception as e:
                print(f"  ✗ {remote_name}: {e}", file=sys.stderr)
                n_fail += 1

    print(f"\nDone. {n_ok} succeeded, {n_fail} failed.", file=sys.stderr)


if __name__ == "__main__":
    main()