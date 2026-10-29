#!/usr/bin/env python
# coding: utf-8
import traceback
import os;
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import sys;sys.path.append("../src")
import json
import torch
import PIL
from pathlib import Path
from openai import OpenAI
from glob import glob
from tqdm import tqdm 
import math
# from utils.prompt_template import PromptTemplate
from datasets import load_dataset, Image
from transformers import Qwen2VLProcessor, AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from datetime import date
import numpy as np
from transformers import GenerationConfig
import transformers;print(transformers.__version__)
import trl;print(trl.__version__)
import torch;print(torch.__version__) # 2.6.0+cu124
import bitsandbytes;print(bitsandbytes.__version__)


# ## Data

# In[ ]:

# ver1
# data_root = "images/collision_diagram/synthetic"
# pickle_file = "dataset" 

# ver2
# data_root = "images/collision_diagram/synthetic_way"
# pickle_file = "dataset_way" 

# data_root = "images/collision_diagram/synthetic_way_hurt"
# pickle_file = "dataset_way_hurt" 

# data_root = "images/collision_diagram/synthetic_mix"
# pickle_file = "dataset_mix" 

# data_root = "images/collision_diagram/synthetic_tune"
# pickle_file = "dataset_tune" 

data_root = "images/collision_diagram/synthetic_tune_v2"
pickle_file = "dataset_tune_v2"

train_dataset, eval_dataset, test_dataset = load_dataset("json",
                    data_files={
                                    "train": [
                                        f"./{data_root}/train/train.jsonl",
                                    ],
                                    "val": [
                                        f"./{data_root}/val/val.jsonl",
                                    ],
                                    "test": [
                                        f"./{data_root}/test/test.jsonl",
                                    ],
                               }
                      , split= ["train", "val", "test"]                         
                 )
print(train_dataset)
train_dataset = train_dataset.filter(lambda x: x['background'] != "")
print(eval_dataset)
train_dataset = train_dataset.cast_column("image", Image())
eval_dataset = eval_dataset.cast_column("image", Image())


# In[ ]:


# User prompt that combines the user query and the schema
user_prompt = "你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖，對路口進行分析並辨識受傷人數和肇事型態。"

# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ],
    }
    
def process_vision_info(messages: list[dict]) -> list[PIL.Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs


# In[ ]:


import pickle
from joblib import dump, load
import time

if os.path.exists(f"data/train_{pickle_file}.pkl"):
    print("✅ 開始讀取 pickle 檔...")

    start_time = time.time()

    with open(f"data/train_{pickle_file}.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    with open(f"data/eval_{pickle_file}.pkl", "rb") as f:
        eval_dataset = pickle.load(f)

    elapsed_time = time.time() - start_time
    print(f"✅ 成功讀取 pickle 檔！耗時 {elapsed_time:.2f} 秒")

else:
    os.makedirs("data", exist_ok=True)
    # Convert dataset to OAI messages
    # train_dataset = train_dataset.select(np.arange(500))
    train_dataset = [format_data(sample) for sample in tqdm(train_dataset)]
    eval_dataset = [format_data(sample) for sample in tqdm(eval_dataset)]

    dump(train_dataset, f"data/train_{pickle_file}.pkl")
    dump(eval_dataset, f"data/eval_{pickle_file}.pkl")
    print("✅ 使用 joblib 加速儲存完成！")

print(train_dataset[345]["messages"])
# # Training

# In[ ]:


import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

today = date.today()
model_id = "Mistral-Small-3.2-24B-Instruct-2506"
model_dir=f"models/{model_id}"
output_dir=f"Output/{model_id}-lora-{today}"

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")


# eager（PyTorch 原生 Attention）
# 這是最基礎的 PyTorch 注意力實作，沒有特別做最佳化，
# 計算過程需要額外的中間張量（attention score、softmax 結果等），GPU 記憶體使用量偏高，速度也相對較慢。

# flash_attention_2（FlashAttention v2）
# 這是專門針對 GPU 記憶體與計算效能最佳化的注意力演算法（尤其是 Ampere 架構或更新的 GPU），
# 它會在計算過程中 減少中間張量的保存，而是直接在計算時進行 online softmax，避免儲存完整的 QK^T 矩陣。

# Define model init arguments
model_kwargs = dict(
    # attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(model_dir, **model_kwargs) # mistralai/Mistral-Small-3.2-24B-Instruct-2506
processor = AutoProcessor.from_pretrained(model_dir) # mistralai/Mistral-Small-3.1-24B-Instruct-2503


# In[9]:


sample_text = processor.apply_chat_template(train_dataset[234]["messages"], tokenize=False, add_generation_prompt=False)
print(sample_text)


# In[10]:


print("Train size:", len(train_dataset))
print("Eval size:", len(eval_dataset))

N = len(train_dataset)
steps_per_epoch = math.ceil(N / 8)
total_steps = 1250 * 3 // 2 
warmup_steps = int(0.08 * 1875) 
print("Total Steps:", total_steps)
print("Warmup Steps:", warmup_steps)

# In[ ]:


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

args = SFTConfig(
    output_dir=output_dir,                      # directory to save and repository id
    num_train_epochs=5,                            # number of training steps
    # max_steps=15000,                            # number of training steps
    per_device_train_batch_size=8,              # batch size per device during training
    gradient_accumulation_steps=2,              # number of steps before performing a backward/update pass
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=100,                         # log every 100 steps
    save_strategy="steps",                      # save checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.5,                          # max gradient norm based on QLoRA paper
    warmup_ratio=0.08,                          # warmup ratio based on QLoRA paper
    lr_scheduler_type="linear",               # use constant learning rate scheduler
    push_to_hub=False,                           # push model to hub
    run_name=output_dir,
    report_to="wandb",                    # report metrics to tensorboard
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
args.remove_unused_columns = False # important for collator


# In[12]:


# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_tokens = [10, 12, 13]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
        
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch


collate_sample = [train_dataset[0], eval_dataset[1]]
collated_data = collate_fn(collate_sample)
# print(collated_data)


# In[ ]:


import wandb
# 金鑰改從 configs/secrets.yaml 讀取（原本寫死在這裡、跟著 commit 進 git 歷史）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fine_tuning.secrets import load_wandb_api_key
wandb.login(key=load_wandb_api_key())


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# start training, the model will be automatically saved to the hub and the output directory
# try:
#     trainer.train()
# except Exception as e:
#     print(f"Exception type: {type(e).__name__}")
#     print(f"Error message: {e}")
#     print("Traceback (most recent call last):")
#     traceback.print_exc()

trainer.train()    
 
# save model 
trainer.save_model(args.output_dir)

# free the memory again
del model
del trainer
torch.cuda.empty_cache()


# # Clear Memory

# In[ ]:


import gc
import time

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
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


# # Merge LoRA adapter in to the original model

# In[ ]:


import shutil

merged_path = f"Output/{model_id}-merged-{today}"

# Load Model base model
model = AutoModelForImageTextToText.from_pretrained(model_dir, low_cpu_mem_usage=True)
model.generation_config.do_sample=True

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_path, safe_serialization=True, max_shard_size="5GB")

# 複製 base tokenizer config 到 merged_path
for fname in [              
                "tekken.json",
                "params.json",
                "generation_config.json",
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "chat_template.json",              
              ]:
    shutil.copy(os.path.join("models/Mistral-Small-3.2-24B-Instruct-2506", fname), 
                os.path.join(merged_path, fname))

processor = AutoProcessor.from_pretrained(output_dir)
processor.save_pretrained(merged_path)

