#!/usr/bin/env python
# coding: utf-8
"""LoRA/SFT 微調 Ministral-3-8B-Instruct-2512，訓練資料為合成的台灣交通事故碰撞構圖。

這是 fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 的簡化複本（2026-09-15 建立，
用 /code-simplification 整理），行為與原本那份逐行一致，差別只有可讀性：把 clear_memory()
裡 5 行重複的 `if "X" in globals(): del globals()["X"]` 收成一個迴圈；把第 9 節「合併 LoRA
adapter 並儲存」原本近 170 行、全部攤平在模組層級的流程拆成
_load_base_model_for_merge() / _save_merged_model() /
_save_merged_model_with_fp8_roundtrip_fix() / _copy_tokenizer_files_to_merged_path()
四個具名函式，模組層級只留下依序呼叫這幾個函式的四行，每個函式內部的技術性註解（FP8
量化細節、已知的 transformers/vLLM 相容性坑）原封不動保留。改完後兩支腳本要各自獨立
維護、互不影響，跟下面這段對 fastmerge.py 本身的說明同理。

fastmerge.py 是 fine_tuning_vlm_trl_traffic_tune_ministral.py 的獨立複本（2026-09-10 建立），跟原本那份
完全分開維護、互不影響。兩者唯一的差異在第 9 節「合併 LoRA adapter 回底模型並儲存」：原本那份
存檔時會先呼叫 merged_model.save_pretrained()，內部的 revert_weight_conversion() 會把已經
dequantize 好的 bf16 權重還原回 FP8 再寫檔，逼得後面得再把整個 ~16GB 的 safetensors 讀回記憶體
修正一次、重寫一次（讀取+重寫實測約 4-5 分鐘）。這份改成自己直接呼叫
transformers.core_model_loading.revert_weight_conversion()（save_pretrained() 內部實際呼叫的
同一個函式）取得「key 命名正確、但被還原回 FP8」的 state_dict，在記憶體裡完成修正後只存檔一次，
省掉「寫錯的版本→整個讀回來→修正→整個重寫」這三段 I/O，只需要真正寫對的版本一次。詳細調查過程
與方案比較見對話記錄（2026-09-10）。

複製自 fine_tuning_vlm_trl_traffic_tune_v2.py（Mistral-Small-3.2-24B/Pixtral 架構）。Ministral-3-8B
跟 Mistral-Small-3.2 是同一個 Mistral3ForConditionalGeneration/PixtralProcessor 架構家族
（image_token_index=10、patch_size=14、spatial_merge_size=2 完全一致），資料前處理與 image token
對齊檢查邏輯不需要另外寫版本，但模型載入方式因為以下原因跟 v2.py 不同，兩支腳本不可互相沿用：
  1. 下載下來的 Instruct checkpoint 本身是 FP8 量化過的（config.json 內嵌 quantization_config，
     quant_method="fp8"），不能像 v2.py 那樣疊加 bitsandbytes 4-bit 量化（會跟已有的 FP8 量化
     衝突）。8.4B 語言模型 + 0.4B 視覺編碼器 bf16 全精度僅需 ~16.6GB，32GB VRAM 完全裝得下，不
     需要任何量化；改成明確傳入 quantization_config=FineGrainedFP8Config(dequantize=True) 讓
     checkpoint 在載入時就地還原成一般 bf16 權重（實測：不傳這個參數的話，模型會悄悄維持 FP8
     量化狀態）。
  2. 拿掉「從 hub 補齊 processor/tokenizer 檔案」的邏輯（本地目錄檔案本身就是齊全的）。
  3. 拿掉手動覆寫 processor.patch_size/spatial_merge_size（本地 processor_config.json 本身就有
     正確值，不像 Mistral-Small 那份本地目錄缺這個檔案）。
  4. AutoProcessor 載入時加上 fix_mistral_regex=True，修正官方已知的 tokenizer regex 問題（不加
     這個參數載入時會印警告）。
  5. 合併階段複製到 merged_path 的 tokenizer 檔案清單改為本地目錄實際擁有的組合（Mistral 系列
     tokenizer 檔 + chat_template.jinja，不是 v2.py 用的 chat_template.json）。
  6. 不需要 bitsandbytes，這支腳本完全不用它。
  7. 模型載入用 dtype=torch.bfloat16，不是 v2.py 的 torch_dtype=torch.bfloat16（transformers
     5.16.1 把 torch_dtype 參數改名成 dtype，torch_dtype 變成已棄用的別名）。
  8. SFTConfig 用 warmup_steps=（從 --warmup_ratio 換算而來），不是 v2.py 的
     warmup_ratio=cli_args.warmup_ratio（transformers 5.16.1 的 TrainingArguments 拿掉了
     warmup_ratio 參數，只剩 warmup_steps；換算邏輯見程式中對應註解）。

詳細背景與實測數據見 docs/superpowers/specs/2026-09-07-ministral-3-8b-fine-tuning-design.md。
"""

import os
# 2026-10-07：改成「外部沒設定才預設用 GPU 0」。Mistral-Small-3.2-24B 搭配 v2.py 的訓練參數
# （batch 8、modules_to_save）單張 H100 80GB 會在第一個 step OOM，需要像 v2.py 一樣用
# CUDA_VISIBLE_DEVICES=0,1 讓 device_map="auto" 把模型切到兩張卡上。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import shutil
import tempfile

import torch
import torch.multiprocessing

# 容器環境的 /dev/shm 常被限制得很小（例如 64MB）。DataLoader 在 num_workers > 0 時預設
# 用 /dev/shm 做 worker 間的張量傳遞（file_descriptor 策略），碰撞構圖的 pixel_values 張量
# 一多就會把 /dev/shm 塞爆，worker 被系統以 Bus error 砍掉（實測跑訓練時真的發生過）。
# 改用 file_system 策略後，張量改透過檔案系統（暫存檔）傳遞，不再受 /dev/shm 大小限制。
torch.multiprocessing.set_sharing_strategy("file_system")


def _assert_fork_start_method() -> None:
    """整支腳本的模型載入與絕大部分程式碼都寫在模組層級（top-level），仰賴 Linux 預設的
    multiprocessing 'fork' 啟動方式：DataLoader worker 是直接 fork 現有 process，共享 fork
    當下已經載入好的物件，不會重新執行整個模組。

    如果 start method 被上游任何套件（或未來的修改）悄悄換成 'spawn'，每個 worker 會改成
    重新 import 這支腳本、從頭把 8B 模型再載入一次——8 個 worker 等於同時多開 8 份模型，
    RAM/VRAM 會在幾秒內被同時吃到好幾倍。這台機器沒有 swap（實測 free -g 顯示 Swap=0），
    一旦 RAM 被這樣吃滿，容易在 OOM killer 介入前先讓整機卡死無回應。這裡在最前面就斷言
    目前仍是 fork，不是的話直接報錯，把問題攔在啟動的第一秒，而不是讓它悄悄把機器拖死。
    """
    start_method = torch.multiprocessing.get_start_method()
    if start_method != "fork":
        raise RuntimeError(
            f"multiprocessing start method={start_method!r}，預期是 'fork'。這支腳本的模型"
            f"載入與大部分程式碼都在模組層級執行，改成 'spawn' 會讓每個 DataLoader worker "
            f"重新執行整個模組、各自載入一次完整模型，記憶體暴增（這台機器沒有 swap，容易"
            f"導致整機無回應）。請勿在別處呼叫 set_start_method('spawn')。"
        )


def _assert_shm_tempdir_is_disk_backed() -> None:
    """file_system 共享策略（見上方 set_sharing_strategy）會把 worker 間傳遞張量的暫存檔案
    寫進 TMPDIR（未設定則預設 /tmp），目的是繞開 /dev/shm（tmpfs，佔用 RAM）容量不足的問題。
    但如果 TMPDIR/tmp 本身也是掛載成 tmpfs/ramfs 之類的記憶體型檔案系統，就完全沒有繞開
    問題，只是換了個一樣會佔用 RAM 的路徑。這台機器目前 /tmp 是磁碟型 overlay（df 實測），
    但這個假設值得在啟動時明確驗證一次，而不是默默假設——這台機器沒有 swap，暫存檔案一旦
    真的佔滿 RAM，會讓整機在 OOM killer 介入前卡死無回應。且崩潰重跑時這批暫存檔案不會被
    清乾淨（實測 /tmp 底下已累積多個 torch-shm-dir-* 殘留目錄），長期執行下去只會越堆越多。
    """
    tempdir = os.path.realpath(tempfile.gettempdir())
    mem_backed_fstypes = {"tmpfs", "ramfs", "shm"}
    best_match = None  # (mount_point 長度, fstype)：取路徑前綴比對到最長的那個掛載點
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fstype = parts[1], parts[2]
                mount_point_norm = mount_point.rstrip("/") or "/"
                if tempdir == mount_point_norm or tempdir.startswith(mount_point_norm + "/") or mount_point_norm == "/":
                    if best_match is None or len(mount_point_norm) > best_match[0]:
                        best_match = (len(mount_point_norm), fstype)
    except OSError as e:
        print(f"⚠️ 無法讀取 /proc/mounts 確認 {tempdir} 的檔案系統型態，略過這項檢查：{e}")
        return

    if best_match is not None and best_match[1] in mem_backed_fstypes:
        raise RuntimeError(
            f"暫存目錄 {tempdir!r} 掛載在記憶體型檔案系統（{best_match[1]}）。"
            f"torch.multiprocessing 的 file_system 共享策略會把 DataLoader worker 間傳遞的"
            f"張量寫進這裡，等於沒有真正避開 /dev/shm 的 RAM 限制。這台機器沒有 swap，訓練"
            f"中把 RAM 塞爆會讓整機無回應。請把 TMPDIR 設到磁碟型檔案系統再重跑。"
        )


_assert_fork_start_method()
_assert_shm_tempdir_is_disk_backed()

from datasets import load_dataset, Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainerCallback, EarlyStoppingCallback
from transformers.trainer_callback import ExportableState
from transformers.utils.quantization_config import FineGrainedFP8Config
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

import transformers
import trl

import sys

# 這支腳本移到 scripts/ 底下後，Python 預設只會把 scripts/ 本身加進 sys.path，
# 找不到專案根目錄的 utils 套件；這裡手動把根目錄加回 sys.path，不論從哪個
# CWD 呼叫（包含 run_formal_finetune.sh 透過 $FINETUNE_SCRIPT 啟動）都能正確 import。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fine_tuning.config import build_arg_parser, training_config_from_args
from utils.fine_tuning.data_formatting import (
    assert_image_token_alignment,
    format_data,
    process_vision_info,
    resolve_image_token_ids,
)
from utils.fine_tuning.resources import detect_max_memory, detect_resources_and_configure
from utils.fine_tuning.run_config_dump import dump_run_configs
from utils.fine_tuning.run_management import find_last_valid_checkpoint, next_and_latest_run_index
from utils.fine_tuning.secrets import load_wandb_api_key

print(f"transformers={transformers.__version__} trl={trl.__version__} torch={torch.__version__}")

if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")


# ============================================================
# 0. 命令列參數
# ============================================================
# --resource_mode 決定 batch size / gradient accumulation / dataloader worker 這三個值怎麼來：
#   auto   -> 用 detect_resources_and_configure() 依目前 GPU/CPU 可用資源自動決定
#   manual -> 直接使用 --batch_size / --accumulation_steps / --num_workers 這三個參數
# （CLI 參數定義與 TrainingConfig 的對應關係見 utils/fine_tuning/config.py）
#
# 2026-10-07：訓練超參數改成跟 fine_tuning_vlm_trl_traffic_tune_v2.py 一致。只在這支腳本覆寫
# parser 預設值，不動 utils/fine_tuning/config.py 的共用 TrainingConfig（fastmerge.py 等其他
# 腳本仍沿用原本的預設值）；CLI 明確傳入的值仍然優先。
#   - warmup_ratio 0.03 -> 0.08、max_grad_norm 0.3 -> 0.5（v2.py 寫死的值）
#   - resource_mode auto -> manual，固定 batch_size=8 × accumulation_steps=2（v2.py 寫死的值；
#     auto 模式會依 GPU 剩餘記憶體改變 batch 配置，32GB 卡上會變成 1 × 16）
#   - early_stopping_patience 15 -> 0：v2.py 沒有 early stopping，固定跑滿 num_train_epochs
#     並以最後一步的權重為準。eval 仍照 --eval_strategy 執行，只用來觀察曲線、不影響訓練結果。
# LoRA 的 modules_to_save 也改成跟 v2.py 一致，見第 6 節。
_arg_parser = build_arg_parser()
_arg_parser.set_defaults(
    warmup_ratio=0.08,
    max_grad_norm=0.5,
    resource_mode="manual",
    batch_size=8,
    accumulation_steps=2,
    early_stopping_patience=0,
    # v2.py 用 flash_attention_2。需要在裝有對應 torch 版本 flash_attn 的環境執行（例如
    # conda env Mistral3-fa，見 scripts/setup_flash_attn_env.sh）；沒有 flash_attn 的環境
    # 請加 --attn_implementation sdpa。
    attn_implementation="flash_attention_2",
)
cli_args = training_config_from_args(_arg_parser.parse_args())


# ============================================================
# 1. 路徑與模型基本設定
# ============================================================
## output_dir 原本用日期命名（today = date.today()）。問題：外部監督迴圈因為 SIGSEGV
# 反覆重跑、且訓練久到跨過午夜時，today 會變成隔天，output_dir 也跟著換了一個全新的
# 空目錄，導致 get_last_checkpoint() 在新目錄裡找不到任何 checkpoint、resume_from_checkpoint
# 悄悄失效，等於從 step 0 整個重來（實測發生過：attempt 3 在 2026-08-31 跑到
# checkpoint-1500，跨午夜後 attempt 4 因為 today 變成 2026-09-01 而完全沒接上）。
# 曾經改成讀環境變數 TRAIN_RUN_DATE 來繞過這個問題，但那需要外部迴圈手動固定日期，
# 且换一天/换一次手動重跑一樣要記得處理。改成用 run index 命名 output_dir：
#   - 預設（不加 --new_run）一律自動接續「目前最大 index」的 run，讀該資料夾內既有
#     checkpoint 續跑 —— 同時涵蓋 SIGSEGV 自動重試、以及使用者自己重新執行這支腳本
#     想接續上次訓練權重這兩種情境，不受日期/跨午夜影響，也不用手動設任何環境變數。
#   - 想開一輪全新訓練（例如換資料集版本、換超參數）才加 --new_run，會另外開
#     run{最大 index + 1} 的新目錄，從 base model 重頭訓練，不會動到舊 run 的資料夾。
model_id = cli_args.model_id
# 2026-10-07：除了 Ministral-3-8B，也支援 Mistral-Small-3.2-24B（同一個 Mistral3/Pixtral 架構家族）。
# 兩者載入方式的差異（FP8 checkpoint、本地缺 processor 檔案）在第 3 節依底模型實際狀態判斷，
# 不靠 model_id 字串分支。
_SUPPORTED_MODEL_IDS = ("Ministral-3-8B-Instruct-2512", "Mistral-Small-3.2-24B-Instruct-2506")
if model_id not in _SUPPORTED_MODEL_IDS:
    raise ValueError(
        f"這支腳本的模型載入邏輯只驗證過 {_SUPPORTED_MODEL_IDS}，收到的 --model_id={model_id!r}。"
        f"跑其他模型請改用對應的腳本（例如 fine_tuning_vlm_trl_traffic_tune_glm.py）。"
    )
model_dir = f"models/{model_id}"

latest_run_index, next_run_index = next_and_latest_run_index(model_id)
run_index = next_run_index if (cli_args.new_run or latest_run_index == 0) else latest_run_index
print(
    f"[run-index] 既有最新 run={latest_run_index or '無'}，"
    f"本次使用 run{run_index}"
    + ("（--new_run 開新一輪）" if cli_args.new_run else "（自動接續）")
)
output_dir = f"Output/{model_id}-lora-run{run_index}"

# Ministral-3-8B-Instruct-2512 從 hub 下載下來的本地目錄本身檔案就是齊全的（config.json /
# processor_config.json / tokenizer_config.json / tokenizer.json / chat_template.jinja），不像
# Mistral-Small 那份本地目錄缺 processor/tokenizer 檔案，這裡不需要 v2.py 那段「從 hub 補齊
# 檔案」的邏輯。


# ============================================================
# 2. 資源自動偵測
#    detect_max_memory()：模型載入前用，決定 device_map="auto" 的 GPU/CPU 記憶體上限
#    detect_resources_and_configure()：訓練參數用，決定 batch size / grad accumulation / worker 數
#    （兩個函式都在 utils/fine_tuning/resources.py）
# ============================================================


# ============================================================
# 3. 載入模型（FP8 checkpoint 還原成 bf16，不量化）與 Processor
# ============================================================
# Ministral-3-8B-Instruct-2512 下載下來的 checkpoint 本身是 FP8 量化過的（config.json 內嵌
# quantization_config，quant_method="fp8"），不像 Mistral-Small-3.2/GLM 是 bf16 原始權重，不能
# 疊加 bitsandbytes 4-bit 量化（會跟已有的 FP8 量化衝突）。8.4B 語言模型 + 0.4B 視覺編碼器 bf16
# 全精度僅需 ~16.6GB（實測），32GB VRAM 完全裝得下，不需要任何量化。這裡明確傳入
# quantization_config=FineGrainedFP8Config(dequantize=True) 讓 checkpoint 在載入時就地還原成
# 一般 bf16 權重（實測：dtype/quantization_config 都不傳的話，模型會悄悄維持 FP8 量化狀態，
# 只佔 ~9.7GB、is_quantized=True；加上 dequantize=True 後才會變成 is_quantized=False 的一般
# bf16 模型，LoRA 可直接套用，不需要 4-bit 訓練的特殊處理）。
#
# attn_implementation 原本寫死 "sdpa"：舊機器（RTX 5090）的 fine_tune_vlm 環境沒有 CUDA
# compiler toolchain（nvcc），flash-attn 連原始碼建置都無法開始（見
# requirements-fine_tune_vlm.txt 旁的安裝紀錄）。現在改由 --attn_implementation 決定，見下方。
#
# 2026-10-07：只有底模型 config.json 本身宣告了 quantization_config（Ministral 的 FP8 checkpoint）
# 才傳 FineGrainedFP8Config(dequantize=True)；Mistral-Small-3.2 是 bf16 原始權重，不需要也不該
# 套用 FP8 設定。
from transformers import AutoConfig

_base_model_config = AutoConfig.from_pretrained(model_dir)
#
# 2026-10-07：attn_implementation 改由 --attn_implementation 決定（這支腳本預設
# flash_attention_2，跟 v2.py 一致）。flash_attn 是要跟 torch 版本/C++ ABI 對應編譯的套件，
# 裝錯版本時 import 會丟 undefined symbol（實測：官方 torch2.8 wheel 裝在 torch 2.13 上），
# 這裡在載入 24B 模型前先 import 一次，把問題攔在啟動的第一秒並給出明確的處理方式。
if cli_args.attn_implementation == "flash_attention_2":
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"--attn_implementation flash_attention_2 但目前的 python 環境無法 import flash_attn：{e}。"
            f"請改用裝有對應 torch 版本 flash_attn 的環境（見 scripts/setup_flash_attn_env.sh），"
            f"或加上 --attn_implementation sdpa。"
        ) from e
    print(f"[attn] 使用 flash_attention_2（flash_attn {flash_attn.__version__}）")
#
# 2026-10-07：多卡時改用明確的 device_map。device_map="auto" 實測把權重切得很平均
# （Mistral-Small 雙卡：GPU0 23.0GB / GPU1 21.7GB，語言模型層 19/21），但它把 lm_head 放在
# GPU0（跟 embed_tokens 同卡），最後一層在 GPU1 算完又把 hidden states 搬回 GPU0 過 lm_head，
# 整份 logits（batch × 序列長度 × 詞表 131072）、loss 計算，加上 modules_to_save 讓
# lm_head/embed_tokens 的可訓練副本、梯度、optimizer state 全部堆在 GPU0。結果訓練峰值
# GPU0 80.9GB / 81.5GB、GPU1 只有 44.9GB（batch 8、flash_attention_2 實測）。
# 這裡把 norm + lm_head 放到最後一張卡，讓 logits 留在最後一張卡上計算，語言模型層再依
# --gpu0_layer_count 切分（不指定則平均切）。單卡時維持 device_map="auto"。
def _build_multi_gpu_device_map(num_layers: int, num_gpus: int, gpu0_layer_count: int | None) -> dict:
    last_gpu = num_gpus - 1
    if gpu0_layer_count is None:
        gpu0_layer_count = num_layers // num_gpus
    if not 0 < gpu0_layer_count < num_layers:
        raise ValueError(f"--gpu0_layer_count 必須介於 1 ~ {num_layers - 1}，收到 {gpu0_layer_count}")
    rest_gpus = num_gpus - 1
    rest_layers = num_layers - gpu0_layer_count
    layer_devices = [0] * gpu0_layer_count + [
        1 + min(i * rest_gpus // rest_layers, rest_gpus - 1) for i in range(rest_layers)
    ]
    device_map = {
        "model.vision_tower": 0,
        "model.multi_modal_projector": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": last_gpu,
        "model.language_model.rotary_emb": last_gpu,
        "lm_head": last_gpu,
    }
    device_map.update({f"model.language_model.layers.{i}": d for i, d in enumerate(layer_devices)})
    return device_map


_num_gpus = torch.cuda.device_count()
if _num_gpus > 1:
    _num_lm_layers = _base_model_config.text_config.num_hidden_layers
    _device_map = _build_multi_gpu_device_map(_num_lm_layers, _num_gpus, cli_args.gpu0_layer_count)
    _layers_per_gpu = [
        sum(1 for name, gpu in _device_map.items() if ".layers." in name and gpu == g) for g in range(_num_gpus)
    ]
    print(f"[device-map] 語言模型 {_num_lm_layers} 層，每張卡層數 {_layers_per_gpu}，lm_head 在 GPU{_num_gpus - 1}")
else:
    _device_map = "auto"

model_kwargs = dict(
    attn_implementation=cli_args.attn_implementation,
    dtype=torch.bfloat16,
    device_map=_device_map,
    max_memory=detect_max_memory(),
)
if getattr(_base_model_config, "quantization_config", None) is not None:
    model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
# Mistral-Small 的 config.json 沒寫頂層 tie_word_embeddings，Mistral3Config 預設 True，但
# text_config 是 False、checkpoint 裡 embed_tokens/lm_head 也是兩份不同的權重（載入時會印
# 「both are present in the checkpoints with different values, so we will NOT tie them」）。
# 頂層的 True 會讓 tie_weights() 在載入後用 torch.equal 比對兩者，lm_head 跟 embed_tokens
# 分在不同卡時直接 RuntimeError（實測：other is on cuda:1 ... wrapper_CUDA__equal）；
# device_map="auto" 也正是因為它把 lm_head 擺回 GPU0。text_config 明確不綁時才覆寫成 False。
if _num_gpus > 1 and getattr(_base_model_config.text_config, "tie_word_embeddings", True) is False:
    model_kwargs["tie_word_embeddings"] = False

model = AutoModelForImageTextToText.from_pretrained(model_dir, **model_kwargs)

# device_map="auto" 在 GPU 可用記憶體不足以裝下整個模型時，會「靜默」把部分層放到 CPU
# （甚至磁碟），不會報錯或印警告。8.4B 語言模型 + 0.4B 視覺編碼器 bf16 全精度僅需 ~16.6GB，
# 32GB VRAM 應該完全裝得下；一旦真的觸發 offload，通常代表當下 GPU 記憶體被別的東西占走
# （例如前一次 attempt 的行程沒有完全釋放），訓練會退化成每個 step 都要在 PCIe 上搬運被
# offload 出去的權重，step 時間可能從 13 秒拉長到數分鐘，且這台機器沒有 swap，offload 出去
# 的權重疊加 activation 記憶體很容易把 RAM 塞爆到整機無回應。與其讓它悄悄變慢、悄悄吃記憶體，
# 這裡直接檢查 hf_device_map，一旦出現 cpu/disk 就立刻報錯，把問題攔在啟動的第一秒。
_offloaded_modules = {
    name: str(device)
    for name, device in getattr(model, "hf_device_map", {}).items()
    if str(device) in ("cpu", "disk")
}
if _offloaded_modules:
    raise RuntimeError(
        f"模型有部分層被 device_map=\"auto\" 悄悄 offload 到 CPU/磁碟：{_offloaded_modules}。"
        f"這通常代表目前 GPU 可用記憶體不足以裝下整個模型（例如有其他行程占用 VRAM），而不是"
        f"這個模型本身裝不下。請先用 nvidia-smi 確認 GPU 記憶體已釋放再重跑，不要讓訓練用"
        f"CPU offload 的方式悄悄跑下去（這台機器沒有 swap，容易把 RAM 塞爆導致整機無回應）。"
    )

# Gradient Checkpointing 訓練時關閉 KV Cache
model.config.use_cache = False
# fix_mistral_regex=True：修正官方回報的 tokenizer regex 已知問題（不加這個參數載入時會印
# 警告，見 https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84）。
#
# Ministral-3-8B 本地目錄的 processor 檔案是齊全的，直接從 model_dir 載入即可。
# Mistral-Small-3.2-24B 的 hub repo（本地目錄同樣）只有 tekken.json/params.json，沒有
# preprocessor_config.json/tokenizer.json/chat_template，AutoProcessor 從本地目錄載入會直接
# OSError。改從同架構、同 tokenizer（vocab 131072、[IMG]/[IMG_BREAK]/[IMG_END]=10/12/13）的
# Mistral-Small-3.1-24B-Instruct-2503 hub repo 取 processor（v2.py 原本的做法，見其載入
# processor 那行的註解），再用底模型 config 的值覆寫 patch_size/spatial_merge_size，確保跟
# 3.2 的 vision_config 一致。
_FALLBACK_PROCESSOR_REPO = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
if os.path.exists(os.path.join(model_dir, "preprocessor_config.json")):
    processor_source = model_dir
else:
    processor_source = _FALLBACK_PROCESSOR_REPO
    print(f"[processor] {model_dir} 沒有 preprocessor_config.json，改從 {processor_source} 載入 processor")
processor = AutoProcessor.from_pretrained(processor_source, fix_mistral_regex=True)
processor.patch_size = _base_model_config.vision_config.patch_size
processor.spatial_merge_size = _base_model_config.spatial_merge_size

# 合成碰撞構圖為 1000x1000，而 preprocessor_config.json 預設 size.longest_edge=1540，
# PixtralImageProcessor 只有在圖片邊長「超過」longest_edge 時才會縮小，1000 < 1540 導致完全
# 沒有被縮圖，每張圖會產生大量 image token、增加訓練時的 GPU 記憶體壓力。這裡把 longest_edge
# 調低，強制縮圖以降低每張圖的 token 數量（仍保留長寬比，並自動對齊 patch_size）。
processor.image_processor.size = {"longest_edge": cli_args.image_longest_edge}


# ============================================================
# 4. 資料集載入（datasets 的 Arrow-backed 格式，圖片延遲解碼）
# ============================================================
# data_root 由 --data_root 指定（各版本對應的 generate* 方法見 --data_root 的 help 說明）
data_root = cli_args.data_root

train_dataset, eval_dataset, test_dataset = load_dataset(
    "json",
    data_files={
        "train": [f"./{data_root}/train/train.jsonl"],
        "val": [f"./{data_root}/val/val.jsonl"],
        "test": [f"./{data_root}/test/test.jsonl"],
    },
    split=["train", "val", "test"],
)
train_dataset = train_dataset.filter(lambda x: x["background"] != "")
train_dataset = train_dataset.cast_column("image", Image())
eval_dataset = eval_dataset.cast_column("image", Image())

print("Train size:", len(train_dataset))
print("Eval size:", len(eval_dataset))


# ============================================================
# 5. 資料轉換：raw sample -> OAI messages -> processor 輸入
#    format_data() / process_vision_info() / assert_image_token_alignment() 都在
#    utils/fine_tuning/data_formatting.py，這裡只組裝需要閉包住 processor/user_prompt 的
#    collate_fn。
# ============================================================
# 注意：這裡刻意不把整個 dataset 轉成 Python list 或用 pickle/joblib 整包快取。舊做法會在
# 訓練開始前，把全部樣本的圖片一次解碼進 RAM（30,000 張 1000x1000 圖片同時存在記憶體中），
# 是造成訓練前就 OOM/記憶體爆炸的主因。現在 train_dataset / eval_dataset 維持 Arrow-backed
# 型態（memory-mapped），圖片轉換（format_data）改到 collate_fn 內執行，DataLoader 每次只會
# 讀取、解碼「當下這一個 batch」需要的樣本。
user_prompt = "你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖，對路口進行分析並辨識受傷人數和肇事型態。"


# ------------------------------------------------------------
# Debug 用：image placeholder token 數量 <-> 圖片尺寸 一致性檢查
# ------------------------------------------------------------
# 拿 processor 實際的 image_ids 屬性，跟下面 labels 遮罩用的硬編碼 [10, 12, 13] 對一次，
# 避免兩邊悄悄不一致（見 utils/fine_tuning/data_formatting.py 的 resolve_image_token_ids()
# 與 assert_image_token_alignment() docstring）。
_IMAGE_TOKEN_IDS = resolve_image_token_ids(processor)


def collate_fn(examples: list) -> dict:
    """資料收集器：把一個 batch 的原始樣本轉成 OAI messages 並用 processor 編碼。

    每個 example 都是「這一批 batch」才從 Arrow-backed dataset 讀出的原始樣本（image 欄位
    到這裡才會真正解碼），format_data 在此才組成 OAI messages，不會提前對整個資料集做這件事。
    """
    texts = []
    images = []
    for example in examples:
        messages = format_data(example, user_prompt=user_prompt)["messages"]
        image_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    assert_image_token_alignment(
        batch,
        examples,
        patch_size=processor.patch_size,
        spatial_merge_size=processor.spatial_merge_size,
        image_token_ids=_IMAGE_TOKEN_IDS,
    )

    # labels 用 input_ids 複製一份，並把 padding token / image token 遮罩掉（不計入 loss）
    labels = batch["input_ids"].clone()
    image_token_ids = [10, 12, 13]
    for image_token_id in image_token_ids:
        labels[labels == image_token_id] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch


# self-test：確認 chat template 與 collate_fn 在正式訓練前可以正常運作
sample_messages = format_data(train_dataset[234], user_prompt=user_prompt)["messages"]
sample_text = processor.apply_chat_template(sample_messages, tokenize=False, add_generation_prompt=False)
print(sample_text)

collate_sample = [train_dataset[0], eval_dataset[1]]
collated_data = collate_fn(collate_sample)


# ============================================================
# 6. LoRA 與訓練參數設定
# ============================================================
peft_config = LoraConfig(
    lora_alpha=cli_args.lora_alpha,
    lora_dropout=cli_args.lora_dropout,
    r=cli_args.lora_r,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    # 2026-10-07：改成跟 v2.py 一致，lm_head/embed_tokens 整個矩陣一起訓練。代價：這兩個矩陣
    # 不小（vocab_size=131072），會變成全精度可訓練參數，adamw 的一階/二階動量會再吃掉數 GB
    # 顯存，每份 checkpoint（含 optimizer state）也會比只存 LoRA adapter 時大上好幾倍
    # （原本約 635MB）。這裡沒有新增任何 special token / 詞彙，不訓練這兩個矩陣也能跑。
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

if cli_args.resource_mode == "auto":
    per_device_train_batch_size, gradient_accumulation_steps, dataloader_num_workers = (
        detect_resources_and_configure(target_effective_batch_size=cli_args.target_effective_batch_size)
    )
else:
    per_device_train_batch_size = cli_args.batch_size
    gradient_accumulation_steps = cli_args.accumulation_steps
    dataloader_num_workers = cli_args.num_workers
    print(
        f"[manual-config] -> per_device_train_batch_size={per_device_train_batch_size}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"dataloader_num_workers={dataloader_num_workers}"
    )

# resource_mode="auto" 時，cli_args.batch_size/accumulation_steps/num_workers 這三個欄位
# 只是 [manual 模式] 用的 CLI 參數、根本沒被拿來用（見 utils/fine_tuning/config.py 的
# --batch_size/--accumulation_steps/--num_workers 說明），detect_resources_and_configure()
# 才是這三個值實際的來源。這裡把偵測出來的有效值回填進 cli_args，讓下面
# dump_run_configs() 存出來的 training_config.yaml 反映「這次啟動實際生效」的數字，而不是
# auto 模式下完全沒被用到、容易誤導人的 CLI 預設值/手動輸入值（實測發生過：auto 模式跑出來
# 的 training_config.yaml 顯示 batch_size=4，但那次其實是偵測到記憶體較小而自動降成 batch_size=1
# 在跑，事後回頭查閱檔案完全看不出來）。
cli_args.batch_size = per_device_train_batch_size
cli_args.accumulation_steps = gradient_accumulation_steps
cli_args.num_workers = dataloader_num_workers

# transformers==5.16.1 的 TrainingArguments（SFTConfig 的父類別）拿掉了 warmup_ratio 參數，只剩
# warmup_steps（實測跑出來的 TypeError）。這裡沿用 --warmup_ratio 這個 CLI 參數的既有語意（三支
# 腳本共用同一組 TrainingConfig，不新增只給這支腳本用的 CLI 參數），在建立 SFTConfig 前先換算成
# 等效的 warmup_steps 整數再傳入；換算公式是 HF Trainer 內部算總 optimizer step 數邏輯的近似值
# （max_steps 優先，否則用 dataset 大小 / effective batch size / epoch 數推算），epoch 數為整數
# 時兩者結果相同，但 --num_train_epochs 若是小數，這裡用的是
# ceil(dataset_size/effective_bs) * ceil(num_train_epochs)，跟 HF 內部
# ceil(num_train_epochs * steps_per_epoch) 在無條件進位的先後順序上有差異，換算出的
# warmup_steps 會略多（例如 2.5 epoch、每 epoch 100 step：這裡算出 300，HF 實際是 250）。
# 預設 num_train_epochs=5（整數）不受影響，僅在手動指定小數 epoch 數時要留意這個落差。
if cli_args.max_steps > 0:
    _total_steps_for_warmup = cli_args.max_steps
else:
    _steps_per_epoch = math.ceil(
        len(train_dataset) / (per_device_train_batch_size * gradient_accumulation_steps)
    )
    _total_steps_for_warmup = _steps_per_epoch * math.ceil(cli_args.num_train_epochs)
warmup_steps = int(cli_args.warmup_ratio * _total_steps_for_warmup)

# 2026-09-14 新增：early stopping —— 連續 --early_stopping_patience 次 eval 沒有比目前最佳
# eval loss 更好（下降超過 --early_stopping_threshold）就停止訓練，預設 15 次。HF 的
# EarlyStoppingCallback 靠 load_best_model_at_end 追蹤的「目前最佳」資訊判斷有沒有進步，
# 要求：(1) 確實有在跑驗證（eval_strategy 不能是 "no"，否則根本沒有 eval loss 可比較）、
# (2) load_best_model_at_end=True 時 save_strategy 必須跟 eval_strategy 完全一致（HF Trainer
# 的 post_init 會直接檢查，這支腳本 save_strategy 固定寫死 "steps"，所以目前只支援
# --eval_strategy steps；--eval_strategy epoch 會跟 save_strategy="steps" 對不上，與其讓 HF
# Trainer 丟出較難懂的內部 ValueError，這裡先攔下來給出明確訊息）。
enable_early_stopping = cli_args.early_stopping_patience > 0
if enable_early_stopping and cli_args.eval_strategy == "no":
    raise ValueError(
        f"--early_stopping_patience={cli_args.early_stopping_patience} > 0 需要搭配 "
        f"--eval_strategy steps 才能判斷 eval loss 有沒有進步（目前 --eval_strategy=no，"
        f"整場訓練不會跑驗證）。請加上 --eval_strategy steps，或用 --early_stopping_patience 0 "
        f"關閉 early stopping。"
    )
if enable_early_stopping and cli_args.eval_strategy != "steps":
    raise ValueError(
        f"--early_stopping_patience={cli_args.early_stopping_patience} > 0 目前只支援 "
        f"--eval_strategy steps（這支腳本 save_strategy 固定是 steps，HF Trainer 要求 "
        f"load_best_model_at_end=True 時兩者一致），收到 --eval_strategy="
        f"{cli_args.eval_strategy!r}。請改用 --eval_strategy steps，或用 "
        f"--early_stopping_patience 0 關閉 early stopping。"
    )


args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=cli_args.num_train_epochs,
    # HF Trainer 的慣例：max_steps > 0 時以 step 數為準（num_train_epochs 被忽略）；
    # max_steps <= 0（預設 -1）則改用 num_train_epochs 決定訓練長度。這裡把 --max_steps
    # 的「0＝不限制」語意換算成 Trainer 認得的 -1，0 以外的正整數則原樣傳入。
    max_steps=cli_args.max_steps if cli_args.max_steps > 0 else -1,
    per_device_train_batch_size=per_device_train_batch_size,  # 依偵測到的 GPU 可用記憶體動態決定
    gradient_accumulation_steps=gradient_accumulation_steps,  # 依 batch size 反推，維持 effective batch size 大致不變
    dataloader_num_workers=dataloader_num_workers,  # 依 CPU 核心數與可用 RAM 動態決定，避免同時 prefetch 太多 batch
    dataloader_prefetch_factor=2 if dataloader_num_workers > 0 else None,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=cli_args.logging_steps,
    save_strategy="steps",
    save_steps=cli_args.save_steps,  # 每幾個 step 存一次 checkpoint（預設沿用 HF 的 500）
    save_total_limit=cli_args.save_total_limit,  # 只保留最新 N 份 checkpoint，避免無限累積吃光磁碟
    # 2026-09-14 新增：原本這裡沒有設定 eval_strategy，SFTTrainer 雖然收到了 eval_dataset，
    # 但 HF Trainer 沒指定 eval_strategy 時預設整場訓練完全不會跑驗證——追加訓練後想確認有沒有
    # overfitting，才發現 log 裡完全沒有 eval 數字可以核對（見對話記錄 2026-09-14）。接上
    # utils/fine_tuning/config.py 新增的 --eval_strategy/--eval_steps，預設每 100 step（跟
    # save_steps 同頻率）評估一次。
    eval_strategy=cli_args.eval_strategy,
    eval_steps=cli_args.eval_steps if cli_args.eval_strategy == "steps" else None,
    # 2026-09-14 新增：early stopping 需要的三個欄位。load_best_model_at_end 只在確認啟用時才
    # 打開（見上方 enable_early_stopping 的推導與檢查），避免 --early_stopping_patience 0（關閉）
    # 時徒增「訓練結束後多存一次最佳 checkpoint」的行為變化；metric_for_best_model/
    # greater_is_better 在 load_best_model_at_end=False 時 HF Trainer 不會使用，維持設定無害。
    load_best_model_at_end=enable_early_stopping,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=cli_args.learning_rate,  # 預設 2e-4，QLoRA 論文建議值
    weight_decay=cli_args.weight_decay,  # 預設 0.0（HF 原生預設值），需要正則化時用 --weight_decay 調整
    bf16=True,
    max_grad_norm=cli_args.max_grad_norm,  # QLoRA 論文建議值
    warmup_steps=warmup_steps,  # 從 --warmup_ratio 換算而來，見上方註解（QLoRA 論文建議 warmup_ratio=0.03）
    lr_scheduler_type=cli_args.lr_scheduler_type,
    push_to_hub=False,
    run_name=output_dir,
    report_to="wandb",
    # 2026-10-07：TRL 1.12 預設 loss_type="chunked_nll"，會 monkey-patch lm_head 的 forward。
    # (1) 模型被 device_map="auto" 切到多張卡時，accelerate 把 forward 換成 functools.partial，
    #     patch 直接丟 AttributeError: 'functools.partial' object has no attribute '__func__'（實測）；
    # (2) chunked 路徑直接讀 lm_head 權重，lm_head 設成 modules_to_save 時有讀不到訓練中那份副本的風險。
    # 改用一般的 "nll"（v2.py 當時的 TRL 版本就是這個行為，數學上相同，只是 logits 峰值記憶體較高）。
    loss_type="nll",
    dataset_text_field="",  # 只是為了滿足 collator 需要一個 dummy 欄位
    dataset_kwargs={"skip_prepare_dataset": True},  # 關閉 TRL 內建的資料集前處理，改用自訂 collate_fn
)
args.remove_unused_columns = False  # 自訂 collate_fn 需要原始欄位（image/text/background），不能被自動移除

# 把這次實際生效的三份設定各自存成 YAML，放進 output_dir 根目錄，事後回頭查一個 run 是用
# 什麼參數跑的不用再翻指令歷史（見 utils/fine_tuning/run_config_dump.py 的說明）。
dump_run_configs(output_dir, training_config=cli_args, sft_args=args, peft_config=peft_config)


# ============================================================
# 7. 建立 Trainer 並訓練
# ============================================================
import wandb

# 2026-09-02 修正：這裡原本直接把一把真的 wandb API key 寫死在程式碼裡並被 commit 進 git
# 歷史（該把外洩的 key 需要另外去 wandb.ai 註銷/輪替，跟這次修正是兩件事）。
# 2026-09-07 再修正：先前改成讀環境變數 WANDB_API_KEY，但每次開新的終端機/被監督迴圈重啟
# 都要記得重新 export，容易漏掉。改成讀 configs/secrets.yaml（不進 git，見
# utils/fine_tuning/secrets.py 的說明），檔案不存在或欄位空白都直接報錯退出，不接受任何
# 寫死在程式碼裡的預設值。
wandb.login(key=load_wandb_api_key())


class _ResyncStepIntervalsAfterResumeCallback(TrainerCallback):
    """修正 HF Trainer resume 時「靜默沿用舊 checkpoint 的存檔/紀錄間隔、忽略這次 CLI 傳入值」
    的行為（追查 --save_steps 傳了新值、但實際還是按舊間隔存檔的成因②，見對話記錄）。

    根因：transformers/trainer.py 的 Trainer._init_training_state() 在 resume 時，會用
    `self.state = TrainerState.load_from_json(checkpoint/trainer_state.json)` 整包蓋掉剛
    用這次 args 算出來的 TrainerState——包含 save_steps/logging_steps/eval_steps 這幾個
    欄位，全部改成「上一次寫這份 checkpoint 時」在用的舊值，不是這次 CLI 傳入的新值。
    之後只有 compare_trainer_and_checkpoint_args() 印一次 `logger.warning_once` 警告
    （這台機器的訓練 log 裡完整出現過：
    `save_steps: 50 (from args) != 10 (from trainer_state.json)`），從頭到尾不會真的把
    值改回來——所以只要這個 run 曾經在某個 save_steps 下存過 checkpoint，之後不管
    --save_steps 帶什麼新值續跑，只要沒有 --new_run，實際存檔間隔永遠卡在那個舊值。

    這個 callback 的 on_train_begin() 剛好卡在 Trainer._init_training_state()（resume 覆蓋
    state 的地方）之後、真正的訓練迴圈開始使用 state.save_steps 之前（見
    transformers/trainer.py 的 _inner_training_loop()：先呼叫 _init_training_state()，
    再呼叫 callback_handler.on_train_begin()，兩者之間沒有任何程式碼會用到這幾個欄位），
    在這裡強制把這三個欄位重新同步回這次真正傳入的 args 值，讓 resume 之後的存檔/紀錄
    間隔跟這次 CLI 參數一致，而不是被舊 checkpoint 的殘留值卡住。
    """

    _SYNCED_ATTRS = ("early_stopping_patience", "save_steps", "logging_steps", "eval_steps")

    def on_train_begin(self, args, state, control, **kwargs):
        mismatches = {}
        for attr in self._SYNCED_ATTRS:
            new_value = getattr(args, attr, None)
            old_value = getattr(state, attr, None)
            if new_value is None:
                continue
            if old_value != new_value:
                mismatches[attr] = (old_value, new_value)
            setattr(state, attr, new_value)
        if mismatches:
            detail = "，".join(
                f"{attr}: checkpoint 殘留值={old} -> 這次 CLI 值={new}"
                for attr, (old, new) in mismatches.items()
            )
            print(f"[resume-fix] resume 後偵測到 TrainerState 跟這次 CLI 參數不一致，已強制同步：{detail}")
        return control


class _BackfillMissingStatefulCallbacksAfterResumeCallback(TrainerCallback):
    """修正 HF Trainer resume 時「舊 checkpoint 的 stateful_callbacks 裡沒有這次新掛上的
    callback」導致下一次存檔直接 KeyError 崩潰的問題（跟上面
    _ResyncStepIntervalsAfterResumeCallback 是同一類「resume 沿用舊 checkpoint 殘留內容、
    忽略這次新狀態」的根因，但這次踩到的是 stateful_callbacks 這個欄位）。

    根因：Trainer._init_training_state() resume 時一樣用
    `self.state = TrainerState.load_from_json(checkpoint/trainer_state.json)` 整包蓋掉
    state，連 stateful_callbacks 這個 dict 也换成「上一次寫這份 checkpoint 時」還存在的那幾個
    callback（例如這個 run 是在加入 --early_stopping_patience 之前就開始訓練的，舊
    checkpoint 的 stateful_callbacks 裡只有 TrainerControl 這個 key，沒有
    EarlyStoppingCallback）。之後第一次呼叫 _save_checkpoint() 存下一份 checkpoint 時，會對
    「這次 trainer 實際掛的每個 ExportableState callback」都執行
    `self.state.stateful_callbacks[cb_name]`（沒有用 .get()，見 transformers/trainer.py
    _save_checkpoint() 該行），字典裡不存在的 key 直接 KeyError、整個訓練程序崩潰——實測
    發生過：從沒有 early stopping 的 checkpoint-600 續跑，這次帶了
    --early_stopping_patience 之後，跑到下一次存檔（step 700）就整個炸掉，且會在每次外部
    監督迴圈重跑時卡在同一個 step 反覆重現（不是隨機硬體中斷，屬於「必定復現」的錯誤）。

    跟 _ResyncStepIntervalsAfterResumeCallback 用同一個時機點：on_train_begin() 卡在
    Trainer._init_training_state()（resume 覆蓋 state 的地方）之後、真正開始訓練/存檔之前，
    在這裡幫每個目前掛著、但 state.stateful_callbacks 裡還沒有對應 key 的 ExportableState
    callback（含 self.control，它本身也是 ExportableState）補上一筆初始值——用該 callback
    「現在」的 state() 當初始值即可，反正這種 callback 是這次訓練才第一次掛上去，本來就沒有
    可以續的歷史狀態可言。
    """

    def __init__(self, callbacks):
        # 用參照而不是在建構當下就複製一份：這個 callback 是在 trainer_callbacks 這個 list
        # 組好之前就先塞進去的，往後才 append 的其他 callback（例如 EarlyStoppingCallback）
        # 需要在 on_train_begin() 真正執行時才看得到，所以要看的是同一個 list 物件本身。
        self._callbacks = callbacks

    def on_train_begin(self, args, state, control, **kwargs):
        candidates = [cb for cb in self._callbacks if isinstance(cb, ExportableState)] + [control]
        backfilled = []
        for cb in candidates:
            cb_name = cb.__class__.__name__
            if cb_name not in state.stateful_callbacks:
                state.stateful_callbacks[cb_name] = cb.state()
                backfilled.append(cb_name)
        if backfilled:
            print(
                "[resume-fix] resume 用的舊 checkpoint 缺少這些 callback 的 "
                f"stateful_callbacks 紀錄（八成是舊 checkpoint 存的時候還沒掛這個 "
                f"callback），已補上初始值：{', '.join(backfilled)}"
            )
        return control


trainer_callbacks = [_ResyncStepIntervalsAfterResumeCallback()]
if enable_early_stopping:
    # early_stopping_patience：連續幾次 eval 沒有比目前最佳 eval loss 進步就停止訓練。
    # early_stopping_threshold：eval loss 至少要下降超過這個量才算「有進步」。兩者都來自
    # --early_stopping_patience/--early_stopping_threshold（見 utils/fine_tuning/config.py），
    # 預設 patience=15、threshold=0.0。
    trainer_callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=cli_args.early_stopping_patience,
            early_stopping_threshold=cli_args.early_stopping_threshold,
        )
    )
    print(
        f"[early-stopping] 啟用：連續 {cli_args.early_stopping_patience} 次 eval "
        f"（每 {cli_args.eval_steps} step 一次）eval loss 沒有下降超過 "
        f"{cli_args.early_stopping_threshold} 就停止訓練"
    )
else:
    print(f"[early-stopping] 停用（--early_stopping_patience={cli_args.early_stopping_patience}）")

# 一定要在 trainer_callbacks 組完（EarlyStoppingCallback 也決定好加不加）之後才 append 這個
# callback：它靠參照讀同一個 list，on_train_begin() 執行當下要看到的是最終版本。
trainer_callbacks.append(_BackfillMissingStatefulCallbacksAfterResumeCallback(trainer_callbacks))

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
    callbacks=trainer_callbacks,
)

# 這台機器上這份腳本已知會在隨機的 step 數被硬體/驅動層級的 SIGSEGV 中斷。在根因（RTX 5090 /
# Blackwell 驅動）修好前，用「偵測 output_dir 內既有 checkpoint、自動從那裡續跑」讓外部的重試
# 迴圈可以一直接著跑，而不是每次崩潰都要從 step 0 重來。
#
# 這裡用自己的 find_last_valid_checkpoint()（utils/fine_tuning/run_management.py）取代
# transformers.trainer_utils.get_last_checkpoint()：後者只認資料夾名稱（checkpoint-<N>
# 裡 N 最大的那個），不管裡面檔案是否寫完整——實測發生過訓練在寫某個 checkpoint 的過程中被
# SIGSEGV 打斷，留下缺 trainer_state.json 的半殘資料夾，導致外部監督迴圈每次重跑都抓到這個
# 壞掉的最新 checkpoint、10 秒內必定 FileNotFoundError，連續衝到 MAX_CONSECUTIVE_FAILURES
# 就放棄重試，完全沒接上其實還完好的上一份 checkpoint。find_last_valid_checkpoint() 會跳過
# 半殘的 checkpoint（改名成 .broken-<timestamp> 隔開，不直接刪除）並改抓上一個完整的。
last_checkpoint = find_last_valid_checkpoint(output_dir)
if last_checkpoint:
    print(f"[resume] 偵測到既有 checkpoint，將從 {last_checkpoint} 繼續訓練")
trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model(args.output_dir)

del model
del trainer
torch.cuda.empty_cache()


# ============================================================
# 8. 訓練後清理記憶體
# ============================================================
def clear_memory():
    for name in ("inputs", "model", "processor", "trainer", "peft_model"):
        globals().pop(name, None)
    time.sleep(2)

    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


clear_memory()

# --skip_merge：訓練完只留 LoRA adapter（trainer.save_model() 已經存到 output_dir 底下），
# 不執行下面第 9 節的合併。給外部流程（例如 scripts/smoke_test_train_merge_vllm.sh）用：
# 合併改交給獨立的 scripts/merge_lora_checkpoint.py 對 output_dir 手動執行，避免這裡跟外部
# 流程各自重新載入底模型 merge 一次，重複讀寫整份 ~16GB 權重。
if cli_args.skip_merge:
    print(f"[skip-merge] --skip_merge，訓練完只保留 LoRA adapter：{output_dir}，不執行內建合併")
    sys.exit(0)


# ============================================================
# 9. 合併 LoRA adapter 回底模型並儲存（fastmerge：省掉寫錯再讀回修正的額外 I/O）
# ============================================================
def _load_base_model_for_merge(model_dir):
    """重新載入底模型，準備跟訓練出來的 LoRA adapter 合併。回傳 (model, base_quant_config)；
    base_quant_config 是底模型原本宣告的量化設定（None 代表底模型本來就不是量化格式），後面
    存檔/複製檔案階段都要靠它判斷該走哪條路徑。

    合併階段重新載入 base model 時，若不傳 quantization_config，checkpoint 會用原本存在磁碟上的
    FP8 量化格式載入，跟訓練出來的 bf16 LoRA adapter 型別對不上，merge_and_unload() 會丟
    `RuntimeError: Promotion for Float8 Types is not supported`（實測跑出來的錯誤）。這裡跟第 3 節
    訓練時載入模型一樣，明確傳 dequantize=True 把 base model 還原成 bf16 再合併。

    重要：FineGrainedFP8Config(dequantize=True) 只傳 dequantize 的話，其餘欄位（尤其
    weight_block_size）會用 dataclass 預設值 (128, 128)。若跟底模型 config.json 實際存的
    quantization_config（Ministral-3-8B-Instruct-2512 是逐張量單一 scale，weight_block_size=
    None，不是逐 128x128 block）對不上，transformers 會靜默放棄套用這個量化設定、把模型當成
    「沒有量化」載入——不會報錯也不會印警告，權重仍保持磁碟上原始的 float8_e4m3fn dtype、數值
    也還是量化後的原始大小（沒有乘上 scale）。第 3 節訓練時的載入（傳了 device_map="auto" +
    max_memory=...）實測不受影響，但這裡的合併重新載入沒有那兩個參數，屬於已知會踩到這個 bug
    的組合（獨立用 merge_lora_checkpoint.py 合併 checkpoint-7800 驗證過：merged 模型權重整組
    是壞的，vLLM 推論時輸出退化成重複同一個控制字元，詳見 2026-09-10 的除錯紀錄）。這裡改用
    AutoConfig 讀出底模型自己宣告的 activation_scheme / weight_block_size /
    modules_to_not_convert 原樣帶入 FineGrainedFP8Config，只覆寫 dequantize=True，讓它跟磁碟上
    實際的量化格式完全對上。
    """
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(model_dir)
    base_quant_config = getattr(base_config, "quantization_config", None)
    merge_model_kwargs = dict(low_cpu_mem_usage=True)
    if base_quant_config is not None:
        merge_model_kwargs["quantization_config"] = FineGrainedFP8Config(
            activation_scheme=base_quant_config.get("activation_scheme", "dynamic"),
            weight_block_size=base_quant_config.get("weight_block_size"),
            modules_to_not_convert=base_quant_config.get("modules_to_not_convert"),
            dequantize=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_dir, **merge_model_kwargs)
    model.generation_config.do_sample = True

    if base_quant_config is not None:
        # 保險檢查：真的 dequantize 成功的話，底模型不該再留下任何 weight_scale_inv /
        # input_scale 這類量化中繼 buffer；如果還在，代表上面的設定又跟某個未來的底模型對不
        # 上，寧可直接中止，也不要把壞掉的權重存成看似正常的 merged 目錄。
        leftover_scale_keys = [
            k for k in model.state_dict() if k.endswith("weight_scale_inv") or k.endswith("input_scale")
        ]
        if leftover_scale_keys:
            raise RuntimeError(
                f"底模型宣告了 quantization_config，但 dequantize 後仍殘留 "
                f"{len(leftover_scale_keys)} 個量化 scale buffer（例如 {leftover_scale_keys[0]}），"
                "代表 dequantize 沒有真的生效，權重可能還是量化前的錯誤數值，已中止避免存出壞掉的模型。"
            )

    return model, base_quant_config


def _save_merged_model_with_fp8_roundtrip_fix(merged_model, merged_path):
    """底模型原本是 FP8 量化格式時，把合併後的 bf16 權重直接存成 bf16（fastmerge 的核心）。

    跟 fine_tuning_vlm_trl_traffic_tune_ministral.py 的差異就在這裡：那份是先讓
    merged_model.save_pretrained() 正常存檔（此時 save_pretrained() 內部會呼叫
    transformers.core_model_loading.revert_weight_conversion() 把 dequantize 好的 bf16
    權重還原回 FP8 + weight_scale_inv 再寫檔），存檔後再整個讀回磁碟修正一次，等於
    對 ~16GB 的資料多做了一輪「寫、讀、再寫」。這裡改成自己先呼叫「save_pretrained()
    內部實際呼叫的同一個函式」revert_weight_conversion()（見 transformers/
    modeling_utils.py 的 `state_dict = revert_weight_conversion(model_to_save,
    state_dict)`），一次拿到「key 命名正確、但被還原回 FP8」的 state_dict，直接在
    記憶體裡修正（乘回 weight_scale_inv、丟掉 scale key），再自己動手分片存檔一次——
    全程只需要真正寫對的版本一次。

    風險：revert_weight_conversion 是 transformers 的內部函式（core_model_loading
    模組，不是穩定公開 API），未來版本可能改名或改變行為。萬一真的失效，下面的
    RuntimeError 保險檢查會直接中止、不會悄悄存出壞掉的模型——失敗模式是「跑不動」，
    不是「跑出錯的東西」。
    """
    print("[fastmerge] 用 revert_weight_conversion() 取得正確命名的 state_dict，"
          "在記憶體裡修正 FP8 round-trip 後一次存檔（不寫錯誤版本、不用讀回磁碟修正）")

    import json

    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file
    from transformers.core_model_loading import revert_weight_conversion

    reverted_state_dict = revert_weight_conversion(merged_model, merged_model.state_dict())

    scales = {k: v for k, v in reverted_state_dict.items() if k.endswith("weight_scale_inv")}
    final_state_dict = {}
    dequantized_count = 0
    for key, tensor in reverted_state_dict.items():
        if key.endswith("weight_scale_inv") or key.endswith("input_scale"):
            continue  # 量化中繼 key，不寫回
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
            scale = scales.get(scale_key)
            if scale is not None:
                tensor = (tensor.float() * scale.float()).to(torch.bfloat16)
                dequantized_count += 1
        final_state_dict[key] = tensor.contiguous()

    print(f"[fastmerge] 已在記憶體裡還原 {dequantized_count} 個權重張量成 bf16"
          "（來源 FP8 + weight_scale_inv），準備存檔")

    # 保險檢查：確認記憶體裡的最終 dict 真的沒有殘留量化 dtype 或量化中繼 key，再動手存檔。
    remaining_bad = [
        k for k, v in final_state_dict.items()
        if k.endswith("weight_scale_inv") or k.endswith("input_scale")
        or v.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ]
    if remaining_bad:
        raise RuntimeError(
            f"修正 FP8 round-trip 後，記憶體裡仍有 {len(remaining_bad)} 個異常 tensor/key"
            f"（例如 {remaining_bad[0]}），已中止，不會存出壞掉的 merged 模型。"
        )

    os.makedirs(merged_path, exist_ok=True)
    merged_model.config.save_pretrained(merged_path)
    merged_model.generation_config.save_pretrained(merged_path)

    split = split_torch_state_dict_into_shards(final_state_dict, max_shard_size="5GB")
    for shard_file, tensor_names in split.filename_to_tensors.items():
        shard = {name: final_state_dict[name] for name in tensor_names}
        save_file(shard, os.path.join(merged_path, shard_file), metadata={"format": "pt"})
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        with open(os.path.join(merged_path, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)


def _save_merged_model(merged_model, merged_path, base_quant_config):
    """把合併後的模型存到 merged_path：底模型本來就沒有量化時直接用標準
    save_pretrained()；底模型本來是 FP8 量化格式時，改走
    _save_merged_model_with_fp8_roundtrip_fix()（見該函式 docstring 說明原因）。
    """
    if base_quant_config is not None:
        _save_merged_model_with_fp8_roundtrip_fix(merged_model, merged_path)
    else:
        merged_model.save_pretrained(merged_path, safe_serialization=True, max_shard_size="5GB")


def _copy_tokenizer_files_to_merged_path(model_dir, merged_path, base_quant_config):
    """複製 base tokenizer config 到 merged_path。

    跟 v2.py 的清單不同：(1) Ministral-3-8B 本地目錄的 chat template 是 chat_template.jinja（不是
    v2.py 用的 chat_template.json），其餘 Mistral 系列 tokenizer 檔案（tekken.json/params.json）
    維持相同；(2) 這裡刻意不複製 config.json——上面已經為合併後的模型寫出正確的 config.json
    （合併後權重是純 bf16，model.config 已經沒有 quantization_config 這個屬性）。若把 config.json
    也列進來複製，會用 model_dir 那份還帶著原始 FP8 quantization_config 的舊 config.json 蓋掉剛
    寫出的正確版本，導致之後重新載入 merged_path 時 transformers 誤以為權重還是 FP8 量化格式，
    丟出 `RuntimeError: getCudnnDataTypeFromScalarType() not supported for Float8_e4m3fn`
    （實測跑出來的錯誤，發生在 vision_tower.patch_conv）。
    """
    for fname in [
        "tekken.json",
        "params.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ]:
        src = os.path.join(model_dir, fname)
        dst = os.path.join(merged_path, fname)
        if not os.path.exists(src):
            # Mistral-Small-3.2 本地目錄沒有 tokenizer_config.json/tokenizer.json/chat_template，
            # 這些檔案由最後的 processor.save_pretrained(merged_path) 補上。
            continue
        if fname == "params.json" and base_quant_config is not None:
            # params.json 是 vLLM `--config_format mistral` 讀的 Mistral 原生設定檔，跟
            # config.json 一樣會內嵌量化宣告，但合併後的權重已經是上面 dequantize 過的 bf16，
            # 不再是 FP8。若原樣複製，vLLM 讀到這個殘留的 quantization 區塊會誤判成 FP8
            # checkpoint 去找對應的 scale tensor，但權重裡早就沒有了。這裡讀進來、拿掉
            # "quantization" 這個 key 再存，其餘欄位（維度、rope 等）不變。
            import json

            with open(src, encoding="utf-8") as f:
                params = json.load(f)
            if params.pop("quantization", None) is not None:
                print(f"[fastmerge] {fname} 移除殘留的量化宣告（合併後權重已是 bf16）")
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
        else:
            shutil.copy(src, dst)


merged_path = f"{output_dir}/merged"

model, base_quant_config = _load_base_model_for_merge(model_dir)
peft_model = PeftModel.from_pretrained(model, output_dir)
merged_model = peft_model.merge_and_unload()
_save_merged_model(merged_model, merged_path, base_quant_config)
_copy_tokenizer_files_to_merged_path(model_dir, merged_path, base_quant_config)

processor = AutoProcessor.from_pretrained(output_dir, fix_mistral_regex=True)
processor.save_pretrained(merged_path)
