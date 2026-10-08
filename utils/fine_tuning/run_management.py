"""訓練輸出目錄（run）與 checkpoint 的管理：決定要接續哪個既有 run，或該從哪個新編號開始；
以及在自動續訓（resume）時，避免抓到存檔中途被中斷、檔案不完整的半殘 checkpoint。
"""

import json
import os
import re
import shutil
from datetime import datetime


def next_and_latest_run_index(model_id: str, output_root: str = "Output") -> tuple:
    """掃描 output_root 底下屬於這個 model_id 的既有 run 資料夾（{model_id}-lora-run<N>），
    回傳 (latest, latest + 1)。latest=0 代表目前還沒有任何既有 run。
    """
    pattern = re.compile(rf"^{re.escape(model_id)}-lora-run(\d+)$")
    indices = []
    if os.path.isdir(output_root):
        for name in os.listdir(output_root):
            match = pattern.match(name)
            if match:
                indices.append(int(match.group(1)))
    latest = max(indices) if indices else 0
    return latest, latest + 1


_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
_TRAINER_STATE_FILE = "trainer_state.json"
# HF Trainer 存 checkpoint 時可能寫出的模型權重檔名——LoRA 訓練是 adapter_model.safetensors；
# 其餘幾種是為了讓這個檢查也能安全套用在非 LoRA（全參數微調）的 checkpoint 上，含 sharded
# 版本的 index 檔。
_WEIGHT_FILE_CANDIDATES = (
    "adapter_model.safetensors",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _is_checkpoint_complete(checkpoint_dir: str) -> bool:
    """檢查一個 checkpoint 資料夾是否完整可用（沒有在存檔中途被中斷）。

    這台機器已知會在隨機 step 數發生硬體/驅動層級的 SIGSEGV/SIGKILL（見
    fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 的說明），若中斷點剛好落在
    Trainer 正在寫某個 checkpoint 的當下，會留下一個「資料夾存在、但檔案沒寫完」的半殘
    checkpoint——trainer_state.json 是存檔流程裡比較晚才寫的檔案，實測最容易缺漏，resume
    時會直接 FileNotFoundError。
    """
    state_path = os.path.join(checkpoint_dir, _TRAINER_STATE_FILE)
    if not os.path.isfile(state_path):
        return False
    try:
        with open(state_path, encoding="utf-8") as f:
            json.load(f)
    except (OSError, json.JSONDecodeError):
        # 檔案存在但讀不出來（例如寫到一半就被砍掉），一樣視為不完整。
        return False
    if not any(
        os.path.isfile(os.path.join(checkpoint_dir, name)) for name in _WEIGHT_FILE_CANDIDATES
    ):
        return False
    return True


def find_last_valid_checkpoint(output_dir: str) -> str | None:
    """取代 transformers.trainer_utils.get_last_checkpoint()：多一層完整性檢查。

    get_last_checkpoint() 只認資料夾名稱（checkpoint-<N> 裡 N 最大的那個），不管裡面檔案
    是否寫完整——實測發生過：訓練在寫 checkpoint-700 的過程中被 SIGSEGV 打斷，留下缺
    trainer_state.json 的半殘資料夾；之後外部監督迴圈（run_formal_finetune.sh）每次自動
    重跑都抓到這個壞掉的 checkpoint-700，10 秒內必定 FileNotFoundError，連續衝到
    MAX_CONSECUTIVE_FAILURES 就放棄重試，完全沒接上其實還完好的 checkpoint-600。

    這裡改成由新到舊依序檢查每個 checkpoint-<N>，跳過不完整的，回傳第一個完整的；全部
    checkpoint 都不完整（或 output_dir 裡根本沒有任何 checkpoint）就回傳 None，讓呼叫端從
    base model 重頭開始訓練。跳過的半殘資料夾會被改名成 `<原名>.broken-<timestamp>`（而不是
    直接刪除，怕誤判、留著方便事後檢查/搶救裡面沒壞的檔案），一來避免它繼續留在原地、下次
    掃描又被撿到，二來避免 HF Trainer 之後做 save_total_limit 輪替清理時被這個假 checkpoint
    混進去計算。
    """
    if not os.path.isdir(output_dir):
        return None

    indices = []
    for name in os.listdir(output_dir):
        match = _CHECKPOINT_DIR_RE.match(name)
        if match:
            indices.append(int(match.group(1)))
    indices.sort(reverse=True)

    for index in indices:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{index}")
        if _is_checkpoint_complete(checkpoint_dir):
            return checkpoint_dir
        quarantined = f"{checkpoint_dir}.broken-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        print(
            f"[resume] {checkpoint_dir} 檔案不完整（疑似存檔中途被中斷），"
            f"改名成 {quarantined} 並改抓上一個 checkpoint 續訓"
        )
        shutil.move(checkpoint_dir, quarantined)

    return None
