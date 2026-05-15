"""
Deploy to Hugging Face Spaces — pure Python, works on Windows.

Usage:
    python spaces/deploy.py --token hf_xxxxx
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SPACE_ID = "soyDCAR/industrial-duct-classifier"
SPACE_FILES = {
    Path("spaces/README.md"): "README.md",
    Path("spaces/app.py"): "app.py",
    Path("spaces/requirements.txt"): "requirements.txt",
    Path("model.py"): "model.py",
}


def deploy(token: str) -> None:
    api = HfApi(token=token)

    # Verify token
    user = api.whoami()["name"]
    print(f"✓ Logged in as: {user}")

    # Create Space if it doesn't exist
    try:
        api.repo_info(repo_id=SPACE_ID, repo_type="space")
        print(f"✓ Space already exists: {SPACE_ID}")
    except Exception:
        create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            token=token,
        )
        print(f"✓ Space created: https://huggingface.co/spaces/{SPACE_ID}")

    # Upload files
    for local_path, remote_name in SPACE_FILES.items():
        if not local_path.exists():
            print(f"  ✗ Missing: {local_path}")
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"chore: update {remote_name}",
        )
        print(f"  ↑ {local_path} → {remote_name}")

    print(f"\n✅ Done → https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="HF write token")
    deploy(p.parse_args().token)
