import json
import sys


def main():
    path = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    only = sys.argv[3] if len(sys.argv) > 3 else None
    shown = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["match"] or (only and r["status"] != only):
                continue
            print(f"[{r['status']}] {r['question']}")
            print(f"  gold: {r['sql']}")
            print(f"  pred: {r['pred']}")
            if r["error"]:
                print(f"  error: {r['error']}")
            shown += 1
            if shown >= count:
                break


if __name__ == "__main__":
    main()
