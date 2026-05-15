#!/usr/bin/env bash
# ── Deploy to Hugging Face Spaces ──────────────────────────────────────
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login          (uses your HF token)
#
# Usage (from the repo root):
#   bash spaces/deploy.sh
#
# On first run: creates the Space and pushes.
# On subsequent runs: pushes updates only.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

HF_USER="soyDCAR"
SPACE_NAME="industrial-duct-classifier"
SPACE_ID="${HF_USER}/${SPACE_NAME}"
CLONE_DIR="/tmp/hf-space-${SPACE_NAME}"

echo "▶ Deploying ${SPACE_ID} to Hugging Face Spaces …"

# 1. Create the Space if it doesn't exist yet
python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
try:
    api.repo_info(repo_id="${SPACE_ID}", repo_type="space")
    print("  Space already exists, skipping creation.")
except Exception:
    api.create_repo(
        repo_id="${SPACE_ID}",
        repo_type="space",
        space_sdk="gradio",
        private=False,
    )
    print("  Space created: https://huggingface.co/spaces/${SPACE_ID}")
EOF

# 2. Clone (or reset) the Space repo
if [ -d "${CLONE_DIR}/.git" ]; then
    echo "▶ Resetting existing clone …"
    git -C "${CLONE_DIR}" fetch origin
    git -C "${CLONE_DIR}" reset --hard origin/main
else
    echo "▶ Cloning Space repo …"
    git clone "https://huggingface.co/spaces/${SPACE_ID}" "${CLONE_DIR}"
fi

# 3. Copy Space files
echo "▶ Copying files …"
cp spaces/README.md       "${CLONE_DIR}/README.md"
cp spaces/app.py          "${CLONE_DIR}/app.py"
cp spaces/requirements.txt "${CLONE_DIR}/requirements.txt"

# model.py is the single dependency from the main repo
cp model.py               "${CLONE_DIR}/model.py"

# 4. Commit and push
cd "${CLONE_DIR}"
git add README.md app.py requirements.txt model.py
git diff --cached --quiet && echo "Nothing to update." && exit 0

git commit -m "chore: sync from soyDCAR/industrial-duct-classifier"
git push

echo ""
echo "✅ Deployed! https://huggingface.co/spaces/${SPACE_ID}"
