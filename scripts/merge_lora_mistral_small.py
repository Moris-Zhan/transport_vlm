#!/usr/bin/env python
# coding: utf-8
"""合併 Mistral-Small-3.2-24B 的 LoRA adapter 回底模型，存成 vLLM 可直接部署的 HF 格式目錄。

照 v2.py 第 9 節的寫法簡化 merge_lora_checkpoint.py（只支援 bf16 的 Mistral-Small，不處理 FP8）。

用法（repo 根目錄，只用 CPU，約 7 分鐘）：
    CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/Mistral3-fa-vllm/bin/python scripts/merge_lora_mistral_small.py \\
      --checkpoint Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run3/checkpoint-100 \\
      --merged_path Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run3/merged

部署：
    VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 vllm serve <merged_path> --tokenizer <merged_path> \\
      --load_format safetensors --tool-call-parser mistral --enable-auto-tool-choice --max-model-len 8192
"""

import argparse
import os

from peft import PeftModel
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

parser = argparse.ArgumentParser(description="合併 Mistral-Small LoRA adapter 回底模型")
parser.add_argument("--checkpoint", required=True, help="LoRA adapter 目錄（run 目錄或其中的 checkpoint-N）")
parser.add_argument("--merged_path", required=True, help="合併後模型輸出目錄")
parser.add_argument("--model_dir", default="models/Mistral-Small-3.2-24B-Instruct-2506", help="底模型目錄")
args = parser.parse_args()

if not os.path.isfile(os.path.join(args.checkpoint, "adapter_config.json")):
    raise FileNotFoundError(f"{args.checkpoint} 底下沒有 adapter_config.json，不是 LoRA checkpoint 目錄")

# Load Model base model
print(f"[merge] 載入底模型（CPU）：{args.model_dir}")
model = AutoModelForImageTextToText.from_pretrained(args.model_dir, low_cpu_mem_usage=True, tie_word_embeddings=False)
model.generation_config.do_sample = True

# Merge LoRA and base model and save
print(f"[merge] 合併 adapter：{args.checkpoint}")
peft_model = PeftModel.from_pretrained(model, args.checkpoint)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(args.merged_path, safe_serialization=True, max_shard_size="5GB")

# 跟 v2.py 不同，刻意不從底模型複製 tekken.json / tokenizer 檔案：processor.save_pretrained()
# 會寫出 HF 的 tokenizer.json；兩者同時存在時 AutoTokenizer 會改用 MistralCommonBackend，
# 把 "[IMG]" 當一般文字切開，vLLM serve 啟動即 image token 數不符（run3 實測）。
processor = AutoProcessor.from_pretrained(args.checkpoint)
# 訓練時 longest_edge 調成 --image_longest_edge（1000），vLLM 卻用 vision_config.image_size（1540）
# 的假圖預估 image token 數，兩邊對不上 serve 就起不來（run3 實測）。改回 1540；碰撞構圖都是
# 1000x1000，兩種設定都不會縮圖，前處理跟訓練時一致。
processor.image_processor.size = {"longest_edge": AutoConfig.from_pretrained(args.model_dir).vision_config.image_size}
processor.save_pretrained(args.merged_path)

print(f"[merge] 完成：{args.merged_path}")
