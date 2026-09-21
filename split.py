import argparse, json, random

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]

    random.seed(args.seed)
    random.shuffle(data)

    test = data[:args.test_size]
    train = data[args.test_size:]

    base = args.input.rsplit(".", 1)[0]
    train_path = f"{base}_train.jsonl"
    test_path = f"{base}_test.jsonl"

    for path, split in [(train_path, train), (test_path, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} → {train_path}")
    print(f"Test:  {len(test)} → {test_path}")

if __name__ == "__main__":
    main()