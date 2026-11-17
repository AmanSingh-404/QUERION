import sys
from pathlib import Path

from huggingface_hub import HfApi

SPACE = "SinghBro/querion"
FOLDER = Path(__file__).resolve().parents[1] / "space"


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else SPACE
    api = HfApi()
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=str(FOLDER),
        repo_id=repo,
        repo_type="space",
        commit_message="Deploy QUERION",
    )
    print(f"https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
