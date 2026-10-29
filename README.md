# transport_vlm — 台灣交通事故碰撞構圖 VLM 微調

用 LoRA / SFT（TRL）微調 Mistral3 家族的視覺語言模型（VLM），讓模型看懂台灣的**肇事碰撞構圖**，
用繁體中文回答路口的**肇事型態、行車方向與傷亡人數**。訓練資料是自行合成的碰撞構圖，合併後的
模型用 vLLM（OpenAI 相容 API）部署，再透過 FastAPI demo 網頁展示。

| 項目 | 內容 |
|---|---|
| 模型 | `Ministral-3-8B-Instruct-2512`（FP8 權重）、`Mistral-Small-3.2-24B-Instruct-2506`（bf16）；`Mistral3ForConditionalGeneration` + `PixtralProcessor` |
| 微調方式 | LoRA + TRL `SFTTrainer`，訓練完可選擇內建合併或另外用合併腳本 |
| 資料 | 以真實路口底圖合成的碰撞構圖（`image` / `text` JSONL），多個格式版本 |
| 部署 | `vllm serve` 合併後模型 → FastAPI demo（`Demo/inference_server.py`） |

---

## 目錄結構

```
transport_vlm/
├── scripts/                     # 所有可執行入口（從 repo 根目錄執行）
│   ├── make_data.py             # 合成資料集（--generator 選格式版本）
│   ├── make_simple_data.py      # make_data.py 的簡化變體
│   ├── fine_tuning_vlm_trl_traffic_tune_ministral_simplified.py   # Ministral-3-8B 訓練
│   ├── fine_tuning_vlm_trl_traffic_tune_mistral_small.py          # Mistral-Small-24B 訓練（雙卡）
│   ├── fine_tuning_vlm_trl_traffic_tune_v2.py                     # 早期的單檔訓練腳本
│   ├── merge_lora_checkpoint.py     # 通用 LoRA 合併（處理 Ministral FP8）
│   ├── merge_lora_mistral_small.py  # Mistral-Small 專用合併（bf16，只用 CPU）
│   ├── run_formal_finetune.sh       # 正式訓練的自動重啟監督迴圈
│   ├── smoke_test_train_merge_vllm.sh  # 訓練 → 合併 → vLLM → 推論 的煙霧測試
│   ├── monitor_progress.sh / monitor.sh  # 訓練進度、GPU 監控
│   ├── setup_flash_attn_env.sh / setup_vllm_env.sh  # 建置 conda 環境
│   └── download_model.py
├── utils/
│   ├── fine_tuning/             # 訓練共用：CLI 參數、資源偵測、run 管理、資料格式、secrets
│   ├── ds_utils/                # 合成資料：底圖、圖示合成、交通規則、各版本 generator
│   ├── accuracy_utils.py        # 模型輸出 vs. ground truth 評分
│   ├── report_utils.py / gen_utils.py / vl_utils.py / utils.py
├── Demo/                        # FastAPI demo 網頁
├── configs/                     # secrets.yaml（wandb key，參考 secrets.yaml.example）
├── images/base_maps/            # 路口底圖與底圖資訊 CSV
├── notebooks/                   # 推論實驗 notebook
├── markdown/                    # 使用說明與交接文件
├── requirements.txt             # 精簡版套件需求
├── requirements-fine_tune_vlm.txt / requirements-Mistral3-fa-vllm.txt  # pip freeze 快照（僅供參考）
└── start.sh                     # 容器部署：vLLM + 前端一起啟動
```

`models/<model_id>/`（底模型）、`data/`（合成資料）、`Output/`（訓練輸出）不在版控內，需自行準備。

---

## 環境

主要套件版本（完整清單見 `requirements.txt`）：

- Python 3.10、torch 2.13.0、transformers 5.16.1、trl 1.12.0、peft 0.20.0
- vllm 0.28.0、fastapi、openai、imagehash、opencv-python-headless

目前的 H100 機器（driver 535，最高支援 CUDA 12.2）使用三層 conda env，每層由上一層 clone 而來：

| env | 內容 | 重建方式 |
|---|---|---|
| `Mistral3` | torch 2.13.0+cu126 + 訓練套件 | — |
| `Mistral3-fa` | + 自行編譯的 flash-attn 2.8.3.post1 | `scripts/setup_flash_attn_env.sh` |
| `Mistral3-fa-vllm` | + vllm 0.28.0+cu129（**訓練與推論共用，建議使用**） | `scripts/setup_vllm_env.sh` |

訓練預設 `--attn_implementation flash_attention_2`；在沒有 flash-attn 的環境請改加
`--attn_implementation sdpa`。

`requirements-*.txt` 中的 freeze 快照只用來參考版本組合，不是拿來 `pip install -r` 的。
上面這兩支 setup 腳本的檔頭說明了為什麼不能直接 `pip install vllm`（CUDA 13 build 在 driver 535 上用不了 GPU）。

### wandb 金鑰

```bash
cp configs/secrets.yaml.example configs/secrets.yaml   # 填入 wandb_api_key
```

本機測試可以設 `WANDB_MODE=offline`，key 隨便填一個字串就好。

---

## 快速開始

以下指令都在 repo 根目錄執行：

```bash
PY=/root/miniconda3/envs/Mistral3-fa-vllm/bin/python
```

### 1. 合成資料集

```bash
$PY scripts/make_data.py --help
$PY scripts/make_data.py --generator CollisionDiagramTuneV2 --dataset_tag tune_v2 \
  --train_num 30000 --val_num 100 --test_num 100
```

輸出到 `data/collision_diagram/synthetic_<dataset_tag>/`。可用的 generator 版本有：

| Generator | 說明 |
|---|---|
| `CollisionDescriptionV1` | 只有路段與肇事類型 |
| `CollisionDescriptionV2` | V1 加上行車方向 |
| `CollisionDescriptionV3` | V2 加上傷亡人數 |
| `CollisionDiagramMix` | 運研所 / 阡陌兩種圖示來源隨機混合 |
| `CollisionDiagramTuneV1` | 加上連續事件疊圖、向量化 IoU 重疊偵測 |
| `CollisionDiagramTuneV2`（預設） | 去重複只看地點，不看主要型態 |

各版本的詳細差異見 [`utils/ds_utils/generators/generators.md`](utils/ds_utils/generators/generators.md)。

### 2. 訓練

**Ministral-3-8B**（`--model_id` 的預設值是 Mistral-Small，所以這裡一定要明確指定）：

```bash
$PY scripts/fine_tuning_vlm_trl_traffic_tune_ministral_simplified.py \
  --model_id Ministral-3-8B-Instruct-2512 \
  --data_root data/collision_diagram/synthetic_tune_v2 \
  --num_train_epochs 10 --eval_strategy steps --eval_steps 100 --new_run
```

**Mistral-Small-3.2-24B**（雙卡；訓練完只存 adapter）：

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline \
  $PY scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py \
  --max_steps 100 --save_steps 50 --eval_steps 50 --new_run
```

常用參數（定義在 `utils/fine_tuning/config.py`）：

- `--new_run`：開一個新的 `run{N+1}`，從底模型開始訓練。不加的話，會從編號最大的 run 裡最後一份**完整** checkpoint 接續訓練。
- `--skip_merge`：只存 LoRA adapter，不做內建合併。
- `--resource_mode auto|manual`：batch size、梯度累積、worker 數由自動偵測決定（`utils/fine_tuning/resources.py`），或由 CLI 參數指定。

訓練輸出寫到 `Output/<model_id>-lora-run<N>/`。

**正式長時間訓練**請用監督迴圈執行。訓練崩潰或卡住超時時它會自動重啟並接續 checkpoint，連續失敗太多次就停止。log 寫到 `formal_finetune_ministral.log`：

```bash
./scripts/run_formal_finetune.sh --model_id Ministral-3-8B-Instruct-2512 --num_train_epochs 10 \
  --eval_strategy steps --eval_steps 100 --data_root data/collision_diagram/synthetic_tune_v2 --new_run

./scripts/monitor_progress.sh -w    # 監控程序、GPU、磁碟、訓練進度
```

### 3. 合併 LoRA

```bash
# Ministral（FP8 底模型）：--model_dir 務必明確指定
$PY scripts/merge_lora_checkpoint.py --checkpoint "$RUN_DIR" --merged_path "$RUN_DIR/merged" \
  --model_dir models/Ministral-3-8B-Instruct-2512

# Mistral-Small（bf16，只用 CPU，約 7 分鐘）
CUDA_VISIBLE_DEVICES="" $PY scripts/merge_lora_mistral_small.py \
  --checkpoint "$RUN_DIR/checkpoint-100" --merged_path "$RUN_DIR/merged"
```

### 4. vLLM 部署

```bash
# Ministral
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$RUN_DIR/merged" \
  --tokenizer_mode mistral --config_format mistral --load_format auto \
  --enable-auto-tool-choice --tool-call-parser mistral --max-model-len 8192 --port 8000

# Mistral-Small
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$RUN_DIR/merged" --tokenizer "$RUN_DIR/merged" \
  --load_format safetensors --enable-auto-tool-choice --tool-call-parser mistral --max-model-len 8192 --port 8000

curl http://localhost:8000/health   # 冷啟動約 3–5 分鐘
```

推論測試的 Python 範例（OpenAI client + base64 圖片）見 [`markdown/USAGE.md`](markdown/USAGE.md) 的 4/4 階段。

### 5. 煙霧測試（一鍵跑完 1→4 步）

```bash
./scripts/smoke_test_train_merge_vllm.sh                          # 跑完會清掉 Output/ 下這次的 run
SMOKE_TEST_KEEP_OUTPUT=1 ./scripts/smoke_test_train_merge_vllm.sh # 保留 run 目錄
```

只訓練 10 step。這個測試驗證的是「訓練 → 合併 → 部署」整條流程能不能跑通，不評估模型效果。
其他可調的環境變數見腳本檔頭。

> 注意：這支腳本與 `USAGE.md` 預設使用 `scripts/fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py`，
> 但這個檔案目前不在 repo 裡。可以先用 `FINETUNE_SCRIPT=scripts/fine_tuning_vlm_trl_traffic_tune_ministral_simplified.py`
> 指定訓練腳本（`_simplified.py` 是 `_fastmerge.py` 的簡化複本，行為相同）。

---

## Demo 網頁

`Demo/inference_server.py` 是一個 FastAPI 應用，會呼叫 `http://localhost:8000/v1` 上的 vLLM。
啟動前請先確認 vLLM 已經在跑，否則 import 階段就會失敗。

```bash
python Demo/inference_server.py --port 8090      # 從 repo 根目錄執行，不要 cd 進 Demo/
```

| 路由 | 說明 |
|---|---|
| `GET /` | 上傳頁 |
| `POST /analyze` | 上傳圖片分析，回 303 跳轉到 `/result/{id}` |
| `GET /result/{result_id}` | 結果頁（查不到 id 回 404） |
| `POST /vlm_response`、`POST /multi_response` | API（可在 `/docs` 的 Swagger 上測試） |

正式容器部署使用 `start.sh`：先啟動 vLLM 並等到 `/health` 就緒，再用 uvicorn 啟動前端（`--root-path` 對應 ingress 前綴）。任一個程序結束時，整個容器都會退出，交給 k8s 重啟。
腳本裡的路徑和模型位置是依部署機器寫死的，換機器時要先修改。

curl 測試範例見 [`markdown/USAGE.md`](markdown/USAGE.md) 後半部。

---

## 已知陷阱（程式碼已處理，請勿「簡化」掉）

- **Ministral 是 FP8 權重**：載入時必須用 `FineGrainedFP8Config(dequantize=True)` 轉成 bf16，而且不可以再疊加 bitsandbytes。合併時 config 的 `weight_block_size` 也必須跟底模型一致，否則 transformers 會默默載入原始 FP8 權重，vLLM 輸出就會變成亂碼。
- **合併後的 `config.json`**：複製 tokenizer 檔案到 `merged/` 時，不可以把底模型的 `config.json` 一起複製過去。那份檔案帶有 FP8 的 `quantization_config`，會蓋掉乾淨的 bf16 config。
- **DataLoader**：容器的 `/dev/shm` 很小，所以 sharing strategy 設為 `file_system`。start method 必須是 `fork`：用 `spawn` 的話，每個 worker 都會重新載入模型，而這台機器沒有 swap。
- **`device_map="auto"`** 可能默默把部分層 offload 到 CPU，腳本裡有檢查這件事。
- **transformers 5.x**：`torch_dtype` 改名為 `dtype`；`warmup_ratio` 已移除，程式會自動換算成 `warmup_steps`。
- **GPU 驅動不穩**：曾經出現隨機 SIGSEGV / SIGKILL 和卡死，所以正式訓練一律透過 `run_formal_finetune.sh` 執行。

---

## 更多文件

- [`markdown/USAGE.md`](markdown/USAGE.md)：煙霧測試的手動四階段流程，以及 Demo 的測試方式
- [`markdown/HANDOFF-2026-10-07-H100訓練.md`](markdown/HANDOFF-2026-10-07-H100訓練.md)：H100 環境建置與 Mistral-Small 微調的交接紀錄
- [`utils/ds_utils/generators/generators.md`](utils/ds_utils/generators/generators.md)：各版本資料 generator 的差異
- [`CLAUDE.md`](CLAUDE.md)：給 Claude Code 的專案說明

## 授權

見 [`LICENSE`](LICENSE)。
