#!/usr/bin/env python
# coding: utf-8
"""Mistral-Small-3.2-24B LoRA/SFT 訓練（雙卡 H100、flash_attention_2）。

由早期的 Ministral-3-8B 訓練腳本（已移除）照 v2.py 的寫法簡化而來：
只支援 Mistral-Small（bf16 權重，不需要 FP8 dequantize）、訓練完只存 LoRA adapter，
合併改用 scripts/merge_lora_mistral_small.py。保留 run 編號與自動接續，
run_formal_finetune.sh 才能在當機後從最後一份完整 checkpoint 接著跑。

用法（repo 根目錄）：
    CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline \\
      /root/miniconda3/envs/Mistral3-fa-vllm/bin/python scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py \\
      --max_steps 100 --save_steps 50 --eval_steps 50 --new_run
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import sys
import tempfile

import torch
import torch.multiprocessing

# 容器的 /dev/shm 很小，DataLoader worker 用預設的 file_descriptor 策略傳 pixel_values 會把它
# 塞爆、worker 被 Bus error 砍掉（實測發生過）。改用 file_system 走 TMPDIR 的暫存檔。
torch.multiprocessing.set_sharing_strategy("file_system")

# 模型載入寫在模組層級，靠 fork 讓 DataLoader worker 共用；換成 spawn 會讓每個 worker 重新載入
# 一次 24B 模型，這台機器沒有 swap，RAM 爆掉會整機卡死。
if torch.multiprocessing.get_start_method() != "fork":
    raise RuntimeError("multiprocessing start method 必須是 fork（spawn 會讓每個 DataLoader worker 重新載入整個模型）")

# file_system 策略的暫存檔寫在 TMPDIR；TMPDIR 若本身是 tmpfs，等於沒有避開 RAM 限制。
_tmpdir = os.path.realpath(tempfile.gettempdir())
_tmpdir_mount, _tmpdir_fstype = max(
    (line.split()[1:3] for line in open("/proc/mounts")
     if _tmpdir.startswith(line.split()[1].rstrip("/") + "/") or _tmpdir == line.split()[1]),
    key=lambda mount: len(mount[0]),
)
if _tmpdir_fstype in ("tmpfs", "ramfs"):
    raise RuntimeError(f"TMPDIR={_tmpdir} 掛在 {_tmpdir_fstype}（{_tmpdir_mount}），請改設到磁碟上再重跑")

from datasets import Image, load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fine_tuning.config import build_arg_parser, training_config_from_args
from utils.fine_tuning.data_formatting import (
    assert_image_token_alignment,
    format_data,
    process_vision_info,
    resolve_image_token_ids,
)
from utils.fine_tuning.run_config_dump import dump_run_configs
from utils.fine_tuning.run_management import find_last_valid_checkpoint, next_and_latest_run_index
from utils.fine_tuning.secrets import load_wandb_api_key

import transformers
import trl

print(f"transformers={transformers.__version__} trl={trl.__version__} torch={torch.__version__}")


# ============================================================
# 0. 參數（預設值與 v2.py 一致；CLI 明確傳值仍優先）
# ============================================================
_arg_parser = build_arg_parser()
_arg_parser.set_defaults(
    model_id="Mistral-Small-3.2-24B-Instruct-2506",
    warmup_ratio=0.08,
    max_grad_norm=0.5,
    batch_size=8,
    accumulation_steps=2,
    attn_implementation="flash_attention_2",
    # 雙卡實測：24/16 層時峰值 GPU0 69.0GB / GPU1 69.4GB（20/20 時 63.8 / 74.4GB）。
    gpu0_layer_count=24,
)
cli_args = training_config_from_args(_arg_parser.parse_args())
if cli_args.model_id != "Mistral-Small-3.2-24B-Instruct-2506":
    raise ValueError(f"這支腳本只支援 Mistral-Small-3.2-24B-Instruct-2506，收到 --model_id={cli_args.model_id!r}")
if cli_args.early_stopping_patience > 0:
    print("⚠️ 這支腳本不支援 early stopping，忽略 --early_stopping_patience")
if cli_args.resource_mode == "auto":
    print("⚠️ 這支腳本固定用 --batch_size/--accumulation_steps/--num_workers，忽略 --resource_mode auto")
if cli_args.skip_merge:
    print("[skip-merge] 這支腳本本來就只存 adapter，合併請用 scripts/merge_lora_mistral_small.py")

model_id = cli_args.model_id
model_dir = f"models/{model_id}"

# output_dir 用 run 編號而不是日期：跨午夜重跑不會換到空目錄。預設接續最新的 run，--new_run 才開新的。
latest_run_index, next_run_index = next_and_latest_run_index(model_id)
run_index = next_run_index if (cli_args.new_run or latest_run_index == 0) else latest_run_index
output_dir = f"Output/{model_id}-lora-run{run_index}"
print(f"[run-index] 本次使用 run{run_index}" + ("（--new_run 開新一輪）" if cli_args.new_run else "（自動接續）"))


# ============================================================
# 1. Data
# ============================================================
data_root = cli_args.data_root
train_dataset, eval_dataset = load_dataset(
    "json",
    data_files={
        "train": [f"./{data_root}/train/train.jsonl"],
        "val": [f"./{data_root}/val/val.jsonl"],
    },
    split=["train", "val"],
)
train_dataset = train_dataset.filter(lambda x: x["background"] != "")
# 維持 Arrow-backed、圖片在 collate_fn 才解碼。v2.py 先把整個資料集轉成 list 再 pickle，
# 30000 張 1000x1000 圖會一次全部解碼進 RAM，是訓練前就 OOM 的主因。
train_dataset = train_dataset.cast_column("image", Image())
eval_dataset = eval_dataset.cast_column("image", Image())
print("Train size:", len(train_dataset))
print("Eval size:", len(eval_dataset))

user_prompt = "你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖，對路口進行分析並辨識受傷人數和肇事型態。"


# ============================================================
# 2. Model & Processor
# ============================================================
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")
if cli_args.attn_implementation == "flash_attention_2":
    # flash_attn 要跟 torch 版本對應編譯，裝錯時 import 會 undefined symbol，先在載入 24B 模型前攔下。
    import flash_attn

    print(f"[attn] 使用 flash_attention_2（flash_attn {flash_attn.__version__}）")

base_config = AutoConfig.from_pretrained(model_dir)
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    raise RuntimeError(f"Mistral-Small batch 8 單卡會在第一個 step OOM，請設 CUDA_VISIBLE_DEVICES=0,1（目前 {num_gpus} 張）")

# 不用 device_map="auto"：它把 lm_head 跟 embed_tokens 一起放在 GPU0，logits/loss 與
# modules_to_save 的梯度、optimizer state 全堆在 GPU0（實測峰值 80.9GB / 81.5GB，GPU1 只有 44.9GB）。
# 這裡把 norm + lm_head 放最後一張卡，語言模型層依 --gpu0_layer_count 切分。
num_layers = base_config.text_config.num_hidden_layers
gpu0_layers = cli_args.gpu0_layer_count or num_layers // num_gpus
if not 0 < gpu0_layers < num_layers:
    raise ValueError(f"--gpu0_layer_count 必須介於 1 ~ {num_layers - 1}，收到 {gpu0_layers}")
rest_layers = num_layers - gpu0_layers
device_map = {
    "model.vision_tower": 0,
    "model.multi_modal_projector": 0,
    "model.language_model.embed_tokens": 0,
    "model.language_model.norm": num_gpus - 1,
    "model.language_model.rotary_emb": num_gpus - 1,
    "lm_head": num_gpus - 1,
    **{f"model.language_model.layers.{i}": 0 for i in range(gpu0_layers)},
    **{
        f"model.language_model.layers.{gpu0_layers + i}": 1 + i * (num_gpus - 1) // rest_layers
        for i in range(rest_layers)
    },
}
print(f"[device-map] 語言模型 {num_layers} 層：GPU0 {gpu0_layers} 層、其餘 {rest_layers} 層；lm_head 在 GPU{num_gpus - 1}")

model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    attn_implementation=cli_args.attn_implementation,
    dtype=torch.bfloat16,
    device_map=device_map,
    # config.json 沒寫頂層 tie_word_embeddings，預設 True；但 text_config 是 False、checkpoint 裡
    # 兩份權重也不同。不覆寫的話 tie_weights() 會用 torch.equal 比對分在兩張卡的
    # lm_head/embed_tokens，直接 RuntimeError（實測）。
    tie_word_embeddings=False,
)
model.config.use_cache = False

# device_map 指定的卡放不下時，accelerate 可能把層丟到 CPU 而不報錯；這台機器沒有 swap，直接攔下。
offloaded = {name: str(dev) for name, dev in model.hf_device_map.items() if str(dev) in ("cpu", "disk")}
if offloaded:
    raise RuntimeError(f"模型有部分層被 offload 到 CPU/磁碟：{offloaded}，請先用 nvidia-smi 確認 GPU 記憶體已釋放")

# Mistral-Small-3.2 本地目錄（與 hub）只有 tekken.json，沒有 processor/tokenizer 檔案，
# 跟 v2.py 一樣改用同架構的 3.1 processor，再用 3.2 的 config 覆寫 patch_size/spatial_merge_size。
processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503", fix_mistral_regex=True)
processor.patch_size = base_config.vision_config.patch_size
processor.spatial_merge_size = base_config.spatial_merge_size
processor.image_processor.size = {"longest_edge": cli_args.image_longest_edge}

image_token_ids = resolve_image_token_ids(processor)


def collate_fn(examples: list) -> dict:
    texts = []
    images = []
    for example in examples:
        messages = format_data(example, user_prompt=user_prompt)["messages"]
        texts.append(processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).strip())
        images.append(process_vision_info(messages))

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    assert_image_token_alignment(
        batch,
        examples,
        patch_size=processor.patch_size,
        spatial_merge_size=processor.spatial_merge_size,
        image_token_ids=image_token_ids,
    )

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()
    for image_token_id in [10, 12, 13]:
        labels[labels == image_token_id] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch


print(processor.apply_chat_template(format_data(train_dataset[234], user_prompt=user_prompt)["messages"], tokenize=False))
collate_fn([train_dataset[0], eval_dataset[1]])


# ============================================================
# 3. Training
# ============================================================
peft_config = LoraConfig(
    lora_alpha=cli_args.lora_alpha,
    lora_dropout=cli_args.lora_dropout,
    r=cli_args.lora_r,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

# transformers 5.x 的 TrainingArguments 拿掉了 warmup_ratio，換算成 warmup_steps。
effective_batch_size = cli_args.batch_size * cli_args.accumulation_steps
if cli_args.max_steps > 0:
    total_steps = cli_args.max_steps
else:
    total_steps = math.ceil(len(train_dataset) / effective_batch_size) * math.ceil(cli_args.num_train_epochs)
warmup_steps = int(cli_args.warmup_ratio * total_steps)
print(f"Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=cli_args.num_train_epochs,
    max_steps=cli_args.max_steps if cli_args.max_steps > 0 else -1,
    per_device_train_batch_size=cli_args.batch_size,
    gradient_accumulation_steps=cli_args.accumulation_steps,
    dataloader_num_workers=cli_args.num_workers,
    dataloader_prefetch_factor=2 if cli_args.num_workers > 0 else None,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=cli_args.logging_steps,
    save_strategy="steps",
    save_steps=cli_args.save_steps,
    save_total_limit=cli_args.save_total_limit,
    eval_strategy=cli_args.eval_strategy,
    eval_steps=cli_args.eval_steps if cli_args.eval_strategy == "steps" else None,
    learning_rate=cli_args.learning_rate,
    weight_decay=cli_args.weight_decay,
    bf16=True,
    max_grad_norm=cli_args.max_grad_norm,
    warmup_steps=warmup_steps,
    lr_scheduler_type=cli_args.lr_scheduler_type,
    push_to_hub=False,
    run_name=output_dir,
    report_to="wandb",
    # TRL 1.12 預設 chunked_nll 會 patch lm_head.forward，模型切到多卡時 accelerate 的
    # functools.partial 會讓它 AttributeError（實測）；用一般 nll（v2.py 當時的行為）。
    loss_type="nll",
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
args.remove_unused_columns = False

dump_run_configs(output_dir, training_config=cli_args, sft_args=args, peft_config=peft_config)


# resume 時 HF Trainer 會用舊 checkpoint 的 trainer_state.json 整包蓋掉 save/logging/eval_steps，
# 這次 CLI 傳的新值被忽略（實測：save_steps 50 (from args) != 10 (from trainer_state.json)）。
# on_train_begin 剛好在覆蓋之後、訓練迴圈使用之前，在這裡改回這次的值。
class ResyncStepIntervalsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        for attr in ("save_steps", "logging_steps", "eval_steps"):
            new_value = getattr(args, attr, None)
            if new_value is not None and getattr(state, attr, None) != new_value:
                print(f"[resume-fix] {attr}: checkpoint 殘留值 {getattr(state, attr, None)} -> {new_value}")
                setattr(state, attr, new_value)
        return control


import wandb

wandb.login(key=load_wandb_api_key())

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
    callbacks=[ResyncStepIntervalsCallback()],
)

# 從最後一份「完整」的 checkpoint 接續；寫到一半被中斷的 checkpoint 會被跳過並改名成 .broken-*。
last_checkpoint = find_last_valid_checkpoint(output_dir)
if last_checkpoint:
    print(f"[resume] 從 {last_checkpoint} 繼續訓練")
trainer.train(resume_from_checkpoint=last_checkpoint)

# save model（adapter + processor）；合併：scripts/merge_lora_mistral_small.py --checkpoint <output_dir 或 checkpoint-N>
trainer.save_model(args.output_dir)
print(f"[done] LoRA adapter 已存至 {output_dir}")
