# 手動流程：訓練 → 合併 → vLLM serve → 推論驗證（Mistral-Small-3.2-24B）

這份文件把 Mistral-Small-3.2-24B 微調後「訓練 → 合併 → 部署 → 驗證」的四個階段拆成可以逐段手動
執行的指令，方便個別除錯、或跳過已知沒問題的階段（例如只想重跑 vLLM serve + 推論測試那兩段）。
指令取自 `scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py`、
`scripts/merge_lora_mistral_small.py` 檔頭的用法說明，以及 `markdown/HANDOFF-2026-10-07-H100訓練.md`。

所有指令都假設在 repo root 底下執行。

## 環境

這台機器（2× H100、driver 535 / CUDA 12.2）上的 conda env 分三層，每層都是從上一層 clone 出來的：

| env | 內容 | 重建方式 |
|---|---|---|
| `Mistral3` | torch 2.13.0+cu126 + 訓練套件（無 flash_attn） | — |
| `Mistral3-fa` | + 自行編譯的 flash-attn 2.8.3.post1 | `scripts/setup_flash_attn_env.sh` |
| `Mistral3-fa-vllm` | + vllm 0.28.0+cu129（訓練與推論都用這個） | `scripts/setup_vllm_env.sh` |

下面四個階段的指令都用 `Mistral3-fa-vllm`。訓練預設 `--attn_implementation flash_attention_2`，
在沒有 flash_attn 的 env 執行訓練時請加 `--attn_implementation sdpa`（實測慢約 1.9 倍）。

```bash
PY=/root/miniconda3/envs/Mistral3-fa-vllm/bin/python
```

## 手動四階段流程

### 1/4 訓練（雙卡，只存 LoRA adapter）

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
  $PY scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py \
  --max_steps 10 \
  --data_root data/collision_diagram/synthetic_tune_v2 \
  --new_run
```

- **Mistral-Small 必須從外部設 `CUDA_VISIBLE_DEVICES=0,1`**，單卡放不下。使用 GPU0 前先 `nvidia-smi`
  確認沒有被外部程序佔用（曾觀察到被本容器看不到的程序佔走約 47GB）。
- 這支腳本只支援 `Mistral-Small-3.2-24B-Instruct-2506`，訓練完用 `trainer.save_model()` 只存 adapter
  與 processor，不做內建合併。
- `--new_run`：開一輪全新訓練，不接續舊 run 的 checkpoint。
- 訓練 log 開頭會印 `[run-index] ... 本次使用 run{N}`，記下這個 N——下面步驟都要用到
  `Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run{N}` 這個目錄（以下用 `RUN_DIR` 代稱）。
- 只跑 10 step 驗證的是流程，不是模型效果；正式長時間訓練請改用 `scripts/run_formal_finetune.sh`。

```bash
RUN_DIR=Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run2   # 換成你實際看到的 run 編號
```

### 2/4 合併（`scripts/merge_lora_mistral_small.py`，只用 CPU，約 7 分鐘）

```bash
CUDA_VISIBLE_DEVICES="" $PY scripts/merge_lora_mistral_small.py \
  --checkpoint "$RUN_DIR" \
  --merged_path "$RUN_DIR/merged"
```

- `--checkpoint` 指向存好 LoRA adapter 的目錄（需含 `adapter_config.json`），可以是 run 目錄本身，
  也可以是中途存下的 `checkpoint-N`。
- `--model_dir` 預設 `models/Mistral-Small-3.2-24B-Instruct-2506`；Mistral-Small 是 bf16 權重，
  不需要處理 FP8。

### 3/4 啟動 vLLM serve

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 \
  /root/miniconda3/envs/Mistral3-fa-vllm/bin/vllm serve "$RUN_DIR/merged" \
  --tokenizer "$RUN_DIR/merged" \
  --load_format safetensors \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --max-model-len 8192 \
  --port 8000
```

另開一個終端機確認就緒（冷啟動實測約 3-5 分鐘）：

```bash
curl http://localhost:8000/health
```

### 4/4 送測試圖片驗證效果

就緒後另開終端機執行：

```bash
$PY - <<'PYEOF'
import base64, json
from io import BytesIO
from openai import OpenAI
from PIL import Image

user_prompt = ("你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。"
               "請基於提供的肇事碰撞構圖，對路口進行分析並辨識受傷人數和肇事型態。")

def encode_image(path):
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model_id = client.models.list().data[0].id

with open("data/collision_diagram/synthetic_tune_v2/test/test.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

resp = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": encode_image(sample["image"])}},
    ]}],
    temperature=0.15, top_p=0.9, max_tokens=1000,
)
print("--- Ground truth ---\n", sample["text"])
print("--- 模型輸出 ---\n", resp.choices[0].message.content)
PYEOF
```

只訓練 10 step 幾乎學不到任何東西，模型輸出不會接近 ground truth 的格式——這個測試驗證的是
「訓練 → 合併 → 部署流程有沒有跑對」，不是模型效果好不好。

## 收尾

手動測完後記得清理，避免佔用 GPU 與磁碟：

```bash
# 關掉 vLLM（Ctrl+C 那個終端機，或找 PID kill 整個 process group）
pkill -f "vllm serve $RUN_DIR/merged"

# 刪掉這次測試產生的 run 目錄
rm -rf "$RUN_DIR"
```

## 相關檔案

- `scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py` — 訓練腳本（只存 LoRA adapter）
- `scripts/merge_lora_mistral_small.py` — 合併腳本，可指定 run 目錄或任一 `checkpoint-N`
- `scripts/run_formal_finetune.sh` — 正式長時間訓練的自動重啟監督迴圈

---

# 手動測試：Demo 網頁（`Demo/inference_server.py`）頁面跳轉

這份記錄 `Demo/inference_server.py` 這個 FastAPI demo 網頁的手動測試方式，涵蓋首頁上傳 →
`/analyze` → 結果頁這條頁面跳轉路徑的輸入輸出介面。

所有指令都假設在 repo root 底下執行——demo 腳本要用
`Demo/inference_server.py` 這個相對路徑啟動，**不要** `cd` 進 `Demo/` 目錄再執行，否則下面
的 curl 範例裡的相對圖片路徑（如 `images/drawio.png`）會對不上。

## 前提

`Demo/inference_server.py` 在 import 階段就會呼叫 `client.models.list()`
（`http://localhost:8000/v1`），所以要先確定有一個 OpenAI 相容的模型服務（例如 vLLM）已經在
本機 8000 port 上跑：

```bash
curl -s http://localhost:8000/v1/models
```

沒有回應或報錯的話，demo server 會在啟動時就直接掛掉。

## 1. 啟動 demo server

```bash
python Demo/inference_server.py --port 8090
```

看到 `Uvicorn running on http://0.0.0.0:8090` 就代表啟動成功。

## 2. 瀏覽器測試（最直觀）

打開 `http://localhost:8090/` → 上傳一張碰撞構圖圖片、送出 → 觀察網址列是否正確跳到
`http://localhost:8090/result/<uuid>` → 檢查結果頁「🔄 分析新圖片」按鈕是否能導回首頁。

## 3. curl 測試（頁面跳轉的輸入輸出介面）

```bash
# 首頁
curl -i http://localhost:8090/

# 送出分析：確認回傳 303，且 Location 是相對路徑 /result/{id}
# （曾經有一版寫死成 Foxconn 正式站絕對網址，已修正）
curl -i -X POST http://localhost:8090/analyze \
  -F "file=@images/drawio.png;type=image/png" \
  -F "prompt=請描述這張碰撞構圖" \
  -F "max_tokens=200"

# 用上一步拿到的 result_id 查結果頁
curl -i http://localhost:8090/result/<result_id>

# 查不存在的 id，應該回 404（快取查無資料的錯誤處理）
curl -i http://localhost:8090/result/not-a-real-id
```

## 4. Swagger 介面（互動測試 `/vlm_response`、`/multi_response` 等 API）

```
http://localhost:8090/docs
```

## 收尾

```bash
pkill -f "inference_server.py --port 8090"
```

## 相關檔案

- `Demo/inference_server.py` — FastAPI demo 主程式，路由：`GET /`、`POST /analyze`、
  `GET /result/{result_id}`、`POST /vlm_response`、`POST /multi_response`
- `Demo/html_templates.py` — 首頁與結果頁的 HTML 模板（`HOME_PAGE_TEMPLATE` /
  `RESULT_PAGE_TEMPLATE`）
- `Demo/test_API.py` — 呼叫正式站部署網址的簡易測試腳本（打的是遠端 API，非本機 demo server）
