#!/bin/bash
# smoke_test_train_merge_vllm.sh —「起飛前檢查」煙霧測試：只訓練少數幾個 step（--max_steps
# --skip_merge），訓練腳本存完 LoRA adapter 就結束、不做內建合併；合併改由獨立的
# scripts/merge_lora_checkpoint.py 對存好的 adapter 目錄執行，再立刻用 vLLM serve 把 merged
# 模型跑起來，送幾張測試圖片驗證「訓練 → 合併 → 部署」整條流程有沒有跑對，不用等正式訓練
# 跑好幾小時才發現某個環節壞掉（FP8 dequantize、殘留 quantization_config、vLLM 啟動參數之類
# 已經踩過的錯誤模式）。
#
# 用法：
#   ./scripts/smoke_test_train_merge_vllm.sh
#
# 環境變數（全部可覆寫，未設定則用預設值）：
#   FINETUNE_PYTHON             python 直譯器路徑，預設 /root/miniconda3/envs/fine_tune_vlm/bin/python
#   FINETUNE_SCRIPT              要跑的訓練腳本，預設
#                                 scripts/fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py
#   MERGE_SCRIPT                  合併 LoRA adapter 的腳本，預設 scripts/merge_lora_checkpoint.py
#   MODEL_DIR                    底模型目錄（合併時傳給 MERGE_SCRIPT 的 --model_dir），預設
#                                 models/Ministral-3-8B-Instruct-2512
#   VLLM_BIN                     vllm CLI 執行檔路徑，預設跟 FINETUNE_PYTHON 同一個 conda env
#                                 底下的 bin/vllm（避免依賴目前這個 shell 有沒有 activate 該 env）
#   SMOKE_TEST_MAX_STEPS         訓練幾個 step，預設 10
#   SMOKE_TEST_NUM_TEST_IMAGES   vLLM 效果測試抽測試集前幾張圖，預設 2
#   SMOKE_TEST_DATA_ROOT          訓練/測試資料目錄，預設 data/collision_diagram/synthetic_tune_v2
#                                 （跟訓練腳本 --data_root 的預設值一致）
#   SMOKE_TEST_KEEP_OUTPUT        設為 1 則測試完不刪除這次產生的 Output/ run 目錄，方便手動
#                                 檢查；預設 0（一律刪除）——smoke test 用的是跟正式訓練共用的
#                                 run-index 編號（next_and_latest_run_index()），留著不刪的話
#                                 會佔掉一個 run 編號、也會佔用 merged 模型的磁碟空間（bf16 全
#                                 精度約 16GB）
#   VLLM_HEALTH_TIMEOUT_SECONDS  vLLM serve 啟動後等待 /health 回應的逾時秒數，預設 600
#                                 （8B 模型冷啟動實測約需數十秒到數分鐘不等）
#   VLLM_PORT                    vLLM serve 監聽的 port，預設 8000
#   SMOKE_TEST_LOG_DIR            訓練 / vLLM 的 log 檔存放目錄，預設 /tmp/smoke_test_train_merge_vllm
#
# 流程：
#   1. 前景執行訓練腳本（--max_steps --skip_merge --new_run），訓練結束只存 LoRA adapter，
#      不做內建合併。從訓練 log 抓出這次用的 run index，組出 run 目錄路徑。
#   2. 前景執行 scripts/merge_lora_checkpoint.py，對 run 目錄手動合併，存到 run 目錄底下的
#      merged/。
#   3. 背景啟動 vLLM serve（指令沿用 notebooks/vlm_infer_real.ipynb 裡已經驗證過、跑得起來的
#      實際組合：--tokenizer_mode/--config_format mistral 等），輪詢 /health 直到就緒或逾時。
#   4. 抽測試集前幾筆圖片送給 vLLM，把模型輸出跟資料集裡的 ground truth 並排印出，供肉眼判斷
#      （10 step 學不到什麼東西，這裡驗證的是「流程有沒有跑對」，不是模型效果好不好）。
#   5. 不論成功或失敗都收尾：關掉 vLLM process；除非 SMOKE_TEST_KEEP_OUTPUT=1，否則刪除這次
#      smoke test 產生的整個 run 目錄。

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

FINETUNE_PYTHON="${FINETUNE_PYTHON:-/root/miniconda3/envs/fine_tune_vlm/bin/python}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-scripts/fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-scripts/merge_lora_checkpoint.py}"
MODEL_ID="Ministral-3-8B-Instruct-2512"
MODEL_DIR="${MODEL_DIR:-models/${MODEL_ID}}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$FINETUNE_PYTHON")/vllm}"
SMOKE_TEST_MAX_STEPS="${SMOKE_TEST_MAX_STEPS:-10}"
SMOKE_TEST_NUM_TEST_IMAGES="${SMOKE_TEST_NUM_TEST_IMAGES:-2}"
SMOKE_TEST_DATA_ROOT="${SMOKE_TEST_DATA_ROOT:-data/collision_diagram/synthetic_tune_v2}"
SMOKE_TEST_KEEP_OUTPUT="${SMOKE_TEST_KEEP_OUTPUT:-0}"
VLLM_HEALTH_TIMEOUT_SECONDS="${VLLM_HEALTH_TIMEOUT_SECONDS:-600}"
VLLM_PORT="${VLLM_PORT:-8000}"
SMOKE_TEST_LOG_DIR="${SMOKE_TEST_LOG_DIR:-/tmp/smoke_test_train_merge_vllm}"

mkdir -p "$SMOKE_TEST_LOG_DIR"
TS="$(date '+%Y%m%d-%H%M%S')"
TRAIN_LOG="$SMOKE_TEST_LOG_DIR/train-${TS}.log"
VLLM_LOG="$SMOKE_TEST_LOG_DIR/vllm-${TS}.log"

echo "=== [smoke-test] 開始 $(date '+%Y-%m-%d %H:%M:%S')，log: $TRAIN_LOG / $VLLM_LOG ==="

# 從訓練 log 裡抓出這次實際用的 run index（訓練腳本會印
# 「[run-index] ... 本次使用 run{N}（--new_run 開新一輪）」這一行，見
# fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 的 run-index 章節）。
extract_run_index() {
    grep -oE '本次使用 run[0-9]+' "$1" | tail -1 | grep -oE '[0-9]+'
}

# --------------------------------------------------------------------------
# 收尾：不論成功、失敗、或被 Ctrl+C 中斷，都要關掉背景的 vLLM process、視情況刪除 run 目錄。
# 用 trap EXIT/INT/TERM 確保一定會執行到，不會留下背景 vLLM process 或殘留的大檔案（merged
# 模型 bf16 全精度約 16GB）。
# --------------------------------------------------------------------------
VLLM_PID=""
RUN_DIR=""
_cleaned_up=0
cleanup() {
    [ "$_cleaned_up" -eq 1 ] && return
    _cleaned_up=1

    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "=== [smoke-test] 關閉 vLLM serve (pid $VLLM_PID) ==="
        kill -TERM -- -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "=== [smoke-test] vLLM 5 秒內沒有回應 SIGTERM，改送 SIGKILL ==="
            kill -KILL -- -"$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
        fi
    fi

    if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
        if [ "$SMOKE_TEST_KEEP_OUTPUT" = "1" ]; then
            echo "=== [smoke-test] SMOKE_TEST_KEEP_OUTPUT=1，保留 $RUN_DIR 供手動檢查 ==="
        else
            echo "=== [smoke-test] 刪除 smoke test 產生的 run 目錄：$RUN_DIR ==="
            rm -rf "$RUN_DIR"
        fi
    fi
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------
# 1. 訓練（--skip_merge：只存 LoRA adapter，不做內建合併，見
#    fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 的 --skip_merge 說明）
# --------------------------------------------------------------------------
echo "=== [smoke-test] 1/4 訓練 ${SMOKE_TEST_MAX_STEPS} step（--skip_merge，只存 LoRA adapter）==="
if ! "$FINETUNE_PYTHON" "$FINETUNE_SCRIPT" \
    --model_id "$MODEL_ID" \
    --max_steps "$SMOKE_TEST_MAX_STEPS" \
    --data_root "$SMOKE_TEST_DATA_ROOT" \
    --skip_merge \
    --new_run \
    > "$TRAIN_LOG" 2>&1; then
    echo "=== [smoke-test] 訓練失敗，log 尾端： ==="
    tail -n 100 "$TRAIN_LOG"
    run_index="$(extract_run_index "$TRAIN_LOG")"
    # 就算訓練途中失敗，output_dir 也可能已經建立，一樣讓 cleanup 清掉，避免留下半成品佔用
    # 磁碟、卡住下次的 run-index。
    [ -n "$run_index" ] && RUN_DIR="Output/${MODEL_ID}-lora-run${run_index}"
    exit 1
fi

run_index="$(extract_run_index "$TRAIN_LOG")"
if [ -z "$run_index" ]; then
    echo "=== [smoke-test] 訓練 exit 0，但無法從 log 找到 run index，log 尾端： ==="
    tail -n 100 "$TRAIN_LOG"
    exit 1
fi
RUN_DIR="Output/${MODEL_ID}-lora-run${run_index}"
MERGED_PATH="$RUN_DIR/merged"

if [ ! -f "$RUN_DIR/adapter_config.json" ]; then
    echo "=== [smoke-test] 訓練 exit 0 但 $RUN_DIR 底下沒有 adapter_config.json，LoRA adapter 疑似沒有存到 ==="
    exit 1
fi
echo "=== [smoke-test] 訓練完成：$RUN_DIR ==="

# --------------------------------------------------------------------------
# 2. 合併 LoRA adapter 回底模型（獨立呼叫 scripts/merge_lora_checkpoint.py，不用訓練腳本內建的
#    合併邏輯）
# --------------------------------------------------------------------------
echo "=== [smoke-test] 2/4 合併 LoRA adapter（scripts/merge_lora_checkpoint.py）==="
if ! "$FINETUNE_PYTHON" "$MERGE_SCRIPT" \
    --checkpoint "$RUN_DIR" \
    --merged_path "$MERGED_PATH" \
    --model_dir "$MODEL_DIR" \
    >> "$TRAIN_LOG" 2>&1; then
    echo "=== [smoke-test] 合併失敗，log 尾端： ==="
    tail -n 100 "$TRAIN_LOG"
    exit 1
fi

if [ ! -f "$MERGED_PATH/config.json" ]; then
    echo "=== [smoke-test] 合併 exit 0 但 $MERGED_PATH 底下沒有 config.json，合併疑似沒有真的完成 ==="
    exit 1
fi
echo "=== [smoke-test] 合併完成：$MERGED_PATH ==="

# --------------------------------------------------------------------------
# 3. 啟動 vLLM serve（指令沿用 notebooks/vlm_infer_real.ipynb 裡已驗證過、跑得起來的組合）
# --------------------------------------------------------------------------
echo "=== [smoke-test] 3/4 啟動 vLLM serve ==="
setsid env VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 \
    "$VLLM_BIN" serve "$MERGED_PATH" \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format auto \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --max-model-len 8192 \
    --port "$VLLM_PORT" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "=== [smoke-test] 等待 vLLM 就緒（最多 ${VLLM_HEALTH_TIMEOUT_SECONDS} 秒，pid $VLLM_PID）==="
waited=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "=== [smoke-test] vLLM process 提早結束，log 尾端： ==="
        tail -n 100 "$VLLM_LOG"
        exit 1
    fi
    if [ "$waited" -ge "$VLLM_HEALTH_TIMEOUT_SECONDS" ]; then
        echo "=== [smoke-test] 等待 vLLM 就緒逾時（${VLLM_HEALTH_TIMEOUT_SECONDS}s），log 尾端： ==="
        tail -n 100 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done
echo "=== [smoke-test] vLLM 就緒（等待約 ${waited} 秒）==="

# --------------------------------------------------------------------------
# 4. 送測試圖片，把模型輸出跟資料集 ground truth 並排印出
# --------------------------------------------------------------------------
echo "=== [smoke-test] 4/4 送測試圖片驗證效果 ==="
"$FINETUNE_PYTHON" - "$SMOKE_TEST_DATA_ROOT" "$SMOKE_TEST_NUM_TEST_IMAGES" "$VLLM_PORT" <<'PYEOF'
import base64
import json
import sys
from io import BytesIO

from openai import OpenAI
from PIL import Image

data_root, num_images, port = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# 跟 fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 第 5 節用的同一句 user_prompt，
# 才是跟訓練資料實際對齊的問法。
user_prompt = (
    "你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。"
    "請基於提供的肇事碰撞構圖，對路口進行分析並辨識受傷人數和肇事型態。"
)


def encode_image(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
model_id = client.models.list().data[0].id
print(f"[smoke-test] vLLM 回報的 model id: {model_id}")

with open(f"{data_root}/test/test.jsonl", encoding="utf-8") as f:
    samples = [json.loads(line) for _, line in zip(range(num_images), f)]

ok = True
for i, sample in enumerate(samples):
    image_path = sample["image"]
    print(f"\n===== 測試樣本 {i + 1}/{len(samples)}：{image_path} =====")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                    ],
                }
            ],
            temperature=0.15,
            top_p=0.9,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"[smoke-test] 送出請求失敗：{e}")
        ok = False
        continue

    print("--- Ground truth ---")
    print(sample["text"])
    print("--- 模型輸出 ---")
    print(answer)

sys.exit(0 if ok else 1)
PYEOF
test_exit_code=$?

if [ "$test_exit_code" -ne 0 ]; then
    echo "=== [smoke-test] 效果測試階段有請求失敗，視為 smoke test 失敗 ==="
    exit 1
fi

echo "=== [smoke-test] 全部完成 $(date '+%Y-%m-%d %H:%M:%S')：訓練 ${SMOKE_TEST_MAX_STEPS} step → 合併 → vLLM serve → 推論測試皆成功 ==="
