# 載入DeepSeek-OCR

from huggingface_hub import login, snapshot_download

# =========================================================
# ========== Hugging Face 連線設定 ==========
# =========================================================
# token 改從 configs/secrets.yaml 的 hf_token 讀取（原本寫死在這裡、跟著 commit 進 git 歷史）。
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fine_tuning.secrets import load_hf_token
TOKEN = load_hf_token()

# HuggingFace 登入
login(token=TOKEN)

import os
import time

def get_dir_size(dir_path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

model_repos = [
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    # "nvidia/Cosmos-Reason2-8B",
    # "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
    # "nvidia/NVLM-D-72B",
    # "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    # "OpenGVLab/InternVL-Chat-V1-5",
    # "OpenGVLab/InternVL3-78B-Instruct",
    # "google/gemma-3-27b-it",
    # "zai-org/GLM-4.6V",
    # "zai-org/GLM-4.6V-Flash",

    # "google/gemma-4-26B-A4B-it",
    # "mistralai/Pixtral-12B-2409",
    # "taide/Gemma-3-TAIDE-12b-Chat-2602",

    # "zai-org/GLM-4.1V-9B-Thinking",
    # "nvidia/Cosmos-Reason2-8B",
    # "mistralai/Ministral-3-8B-Instruct-2512"
]

total_size = 0

for idx, model_repo in enumerate(model_repos, 1):
    print(f"({idx}/{len(model_repos)}) 下載 {model_repo}...")
    t1 = time.time()
    # local_dir = f"models/{model_repo.replace('/', '_')}"
    local_dir = f"models/{model_repo.split('/', 1)[-1]}"
    snapshot_download(
        repo_id=model_repo,   
        revision="main",  # 或指定具體 commit hash
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=TOKEN,
        resume_download=True
    )
    dl_size = get_dir_size(local_dir)
    total_size += dl_size
    elapsed = time.time() - t1
    print(f"→ {model_repo} 下載完成, 大小: {dl_size/1024/1024/1024:.2f} GB, 花費 {elapsed:.1f} 秒\n")

print(f"全部模型下載完成，總佔用空間: {total_size/1024/1024/1024:.2f} GB")

# google/gemma-4-26B-A4B-it 下載完成, 大小: 48.10 GB
# mistralai/Pixtral-12B-2409 下載完成, 大小: 23.64 GB
# taide/Gemma-3-TAIDE-12b-Chat-2602 下載完成, 大小: 25.43 GB