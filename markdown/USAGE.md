# 手動測試：訓練 → 合併 → vLLM serve → 推論驗證

這份文件記錄 `scripts/smoke_test_train_merge_vllm.sh`（Ministral-3-8B 微調的「起飛前檢查」煙霧
測試）內部實際執行的四個階段，拆成可以逐段手動執行的指令，方便個別除錯、或跳過已知沒問題
的階段（例如訓練+合併很花時間，只想重跑 vLLM serve + 推論測試那兩段）。

所有指令都假設在 repo root（`transport_vlm_refactor/`）底下執行。

## 自動化版本

不需要手動分段時，直接跑整支腳本即可，四個階段會依序自動執行，成功或失敗都會清掉這次產生
的 `Output/` run 目錄（含 ~16GB 的 merged 模型）：

```bash
./scripts/smoke_test_train_merge_vllm.sh
```

若想跑完後保留 run 目錄（例如接下來要手動重跑 3/4、4/4 階段），可以：

```bash
SMOKE_TEST_KEEP_OUTPUT=1 ./scripts/smoke_test_train_merge_vllm.sh
```

腳本支援的其他環境變數（訓練 step 數、測試圖片數量、port、逾時秒數等）見
`scripts/smoke_test_train_merge_vllm.sh` 檔頭註解。

## flash-attn 訓練環境

訓練腳本預設 `--attn_implementation flash_attention_2`，需要裝有對應 torch 版本 flash_attn 的
conda env。這台機器（H100、driver 535 / CUDA 12.2）上的 env 分三層，每層都是從上一層 clone 出來的：

| env | 內容 | 重建方式 |
|---|---|---|
| `Mistral3` | torch 2.13.0+cu126 + 訓練套件（無 flash_attn） | — |
| `Mistral3-fa` | + 自行編譯的 flash-attn 2.8.3.post1 | `scripts/setup_flash_attn_env.sh` |
| `Mistral3-fa-vllm` | + vllm 0.28.0+cu129（訓練與推論都用這個） | `scripts/setup_vllm_env.sh` |

下面四個階段的指令都用 `Mistral3-fa-vllm`。在沒有 flash_attn 的 env 執行訓練時請加
`--attn_implementation sdpa`。

## 手動四階段流程

### 1/4 訓練（只存 LoRA adapter，不做內建合併）

```bash
/root/miniconda3/envs/Mistral3-fa-vllm/bin/python \
  scripts/fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py \
  --model_id Ministral-3-8B-Instruct-2512 \
  --max_steps 10 \
  --data_root data/collision_diagram/synthetic_tune_v2 \
  --skip_merge \
  --new_run
```

- `--skip_merge`：訓練完只存 LoRA adapter（`trainer.save_model()` 存到 output_dir），不執行
  腳本內建的合併（見 `utils/fine_tuning/config.py` 的 `--skip_merge` 說明）。
- `--new_run`：開一輪全新訓練，不接續舊 run 的 checkpoint。
- 訓練 log 開頭會印 `[run-index] ... 本次使用 run{N}`，記下這個 N——下面步驟都要用到
  `Output/Ministral-3-8B-Instruct-2512-lora-run{N}` 這個目錄（以下用 `RUN_DIR` 代稱）。

```bash
RUN_DIR=Output/Ministral-3-8B-Instruct-2512-lora-run2   # 換成你實際看到的 run 編號
```

### 2/4 合併（`scripts/merge_lora_checkpoint.py`）

```bash
/root/miniconda3/envs/Mistral3-fa-vllm/bin/python \
  scripts/merge_lora_checkpoint.py \
  --checkpoint "$RUN_DIR" \
  --merged_path "$RUN_DIR/merged" \
  --model_dir models/Ministral-3-8B-Instruct-2512
```

- `--checkpoint` 指向 1/4 步驟存好 LoRA adapter 的目錄（需含 `adapter_config.json`）。
- `--model_dir` **務必明確指定**：這支腳本是通用的合併工具，可以合併不同模型家族
  （Mistral-Small/GLM/Ministral）的 checkpoint，`--model_dir` 預設值不一定對應你要合併的
  模型，記得換成對應的底模型目錄。
- 合併邏輯跟 `fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py` 第 9 節一致：在記憶體
  裡用 `revert_weight_conversion()` 修正 FP8 round-trip 後一次存檔，log 會印
  `[merge] 用 revert_weight_conversion() 取得正確命名的 state_dict...`。

### 3/4 啟動 vLLM serve

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 \
  /root/miniconda3/envs/Mistral3-fa-vllm/bin/vllm serve "$RUN_DIR/merged" \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format auto \
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
/root/miniconda3/envs/Mistral3-fa-vllm/bin/python - <<'EOF'
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
EOF
```

只訓練 10 step 幾乎學不到任何東西，模型輸出不會接近 ground truth 的格式——這個測試驗證的是
「訓練 → 合併 → 部署流程有沒有跑對」，不是模型效果好不好。

## 收尾

手動測完後記得清理，避免佔用 GPU 與磁碟（merged 模型 bf16 全精度約 16GB）：

```bash
# 關掉 vLLM（Ctrl+C 那個終端機，或找 PID kill 整個 process group）
pkill -f "vllm serve $RUN_DIR/merged"

# 刪掉這次測試產生的 run 目錄
rm -rf "$RUN_DIR"
```

## 相關檔案

- `scripts/smoke_test_train_merge_vllm.sh` — 把以上四階段自動串起來的煙霧測試腳本
- `scripts/fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py` — 訓練腳本（第 9 節是
  內建合併邏輯，`--skip_merge` 可跳過）
- `scripts/merge_lora_checkpoint.py` — 獨立合併腳本，用於手動合併任意 LoRA checkpoint（例如
  訓練中途中斷、只存到 `checkpoint-N` 而非最終路徑時）

---

# 手動測試：Demo 網頁（`demo/inference_server.py`）頁面跳轉

這份記錄 `demo/inference_server.py` 這個 FastAPI demo 網頁的手動測試方式，涵蓋首頁上傳 →
`/analyze` → 結果頁這條頁面跳轉路徑的輸入輸出介面。

所有指令都假設在 repo root（`transport_vlm_refactor/`）底下執行——demo 腳本要用
`demo/inference_server.py` 這個相對路徑啟動，**不要** `cd` 進 `demo/` 目錄再執行，否則下面
的 curl 範例裡的相對圖片路徑（如 `images/drawio.png`）會對不上。

## 前提

`demo/inference_server.py` 在 import 階段就會呼叫 `client.models.list()`
（`http://localhost:8000/v1`），所以要先確定有一個 OpenAI 相容的模型服務（例如 vLLM）已經在
本機 8000 port 上跑：

```bash
curl -s http://localhost:8000/v1/models
```

沒有回應或報錯的話，demo server 會在啟動時就直接掛掉。

## 1. 啟動 demo server

```bash
cd /mnt/moris/Dcoker_Project/Project/transport_vlm_refactor
python demo/inference_server.py --port 8090
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

- `demo/inference_server.py` — FastAPI demo 主程式，路由：`GET /`、`POST /analyze`、
  `GET /result/{result_id}`、`POST /vlm_response`、`POST /multi_response`
- `demo/html_templates.py` — 首頁與結果頁的 HTML 模板（`HOME_PAGE_TEMPLATE` /
  `RESULT_PAGE_TEMPLATE`）
- `demo/test_API.py` — 呼叫正式站部署網址的簡易測試腳本（打的是遠端 API，非本機 demo server）
