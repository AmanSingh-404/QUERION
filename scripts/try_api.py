import sys

import httpx


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
    question = sys.argv[2] if len(sys.argv) > 2 else "How many singers are older than 30?"
    body = {
        "tables": ["CREATE TABLE singer (singer_id INTEGER, name TEXT, country TEXT, age INTEGER)"],
        "question": question,
        "samples": 4,
    }
    with httpx.stream("POST", f"{url}/generate", json=body, timeout=120) as response:
        print("status", response.status_code)
        for line in response.iter_lines():
            if line:
                print(line)


if __name__ == "__main__":
    main()
