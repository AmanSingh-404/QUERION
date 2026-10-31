import sys

from huggingface_hub import HfApi


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-release/querion-100m-sql"
    repo = sys.argv[2] if len(sys.argv) > 2 else "SinghBro/querion-100m-sql"
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=folder,
        repo_id=repo,
        repo_type="model",
        commit_message="QUERION v0.6.0",
    )
    print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
