#!/usr/bin/env python
# coding: utf-8
"""合併指定的 LoRA adapter checkpoint 回底模型，儲存為獨立可部署的模型目錄。

沿用 fine_tuning_vlm_trl_traffic_tune.py 第 9 節「合併 LoRA adapter 回底模型並儲存」的
邏輯，抽成獨立腳本：訓練因故（例如已知的 RTX 5090 SIGSEGV，見
markdown/SEGFAULT_RTX5090.md）中途停止、只存到某個 checkpoint-N 而非
trainer.save_model(output_dir) 的「最終」路徑時，用這支腳本手動指定要合併哪個
checkpoint 子目錄。

用法（在 repo root 底下執行；--checkpoint / --merged_path 仍是相對於 repo root 的路徑）：
    python scripts/merge_lora_checkpoint.py \
        --checkpoint Output/Mistral-Small-3.2-24B-Instruct-2506-lora-2026-08-31/checkpoint-2000 \
        --merged_path Output/Mistral-Small-3.2-24B-Instruct-2506-merged-2026-08-31-ckpt2000

        python scripts/merge_lora_checkpoint.py \
        --checkpoint Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run1/checkpoint-7800 \
        --merged_path Output/Mistral-Small-3.2-24B-Instruct-2506-merged-2026-08-31-ckpt7800

        python scripts/merge_lora_checkpoint.py \
        --checkpoint Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run3/checkpoint-11800 \
        --merged_path Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run3/merged-ckpt11800
"""

import argparse
import os
import shutil

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.utils.quantization_config import FineGrainedFP8Config
from peft import PeftModel

parser = argparse.ArgumentParser(description="合併 LoRA adapter checkpoint 回底模型")
parser.add_argument("--checkpoint", type=str, required=True, help="LoRA adapter checkpoint 目錄（含 adapter_config.json / adapter_model.safetensors）")
parser.add_argument("--merged_path", type=str, required=True, help="合併後模型輸出目錄")
parser.add_argument(
    "--model_dir",
    type=str,
    default="models/Mistral-Small-3.2-24B-Instruct-2506",
    help="底模型目錄（未量化，供合併用）",
)
parser.add_argument(
    "--copy_tekken",
    action="store_true",
    help="一併複製底模型的 tekken.json（只有 vLLM 用 --tokenizer_mode mistral 部署時才需要）",
)
args = parser.parse_args()

if not os.path.isfile(os.path.join(args.checkpoint, "adapter_config.json")):
    raise FileNotFoundError(f"{args.checkpoint} 底下沒有 adapter_config.json，確認路徑是否為正確的 LoRA checkpoint 目錄")

print(f"[merge] 載入底模型（未量化，CPU/RAM）：{args.model_dir}")
# 有些底模型目錄本身是量化過的 checkpoint（例如 Mistral-Small-3.2-24B-Instruct-2506 的 FP8
# checkpoint，config.json 內嵌 quantization_config）。若直接照原本的方式載入（不還原量化），
# 合併階段 merge_and_unload() 會因為 base 權重跟 bf16 的 LoRA adapter 型別對不上，丟出
# `RuntimeError: Promotion for Float8 Types is not supported`（fine_tuning_vlm_trl_traffic_
# tune_ministral.py 實測跑出來的錯誤，見 commit e916db3）。這裡用 AutoConfig 檢查底模型是否
# 內嵌了 quantization_config，有的話才明確傳 dequantize=True 還原成一般精度再合併；bf16
# 原生的底模型（Mistral-Small/GLM）沒有這個欄位，維持原本的載入方式不變。
#
# 重要：FineGrainedFP8Config(dequantize=True) 只傳 dequantize 的話，其餘欄位（尤其
# weight_block_size）會用 dataclass 預設值 (128, 128)，若跟底模型 config.json 實際存的
# quantization_config（例如 Mistral-Small-3.2-24B-Instruct-2506 是逐張量單一 scale，
# weight_block_size=None，不是逐 128x128 block）對不上，transformers 會整個放棄套用
# 這個量化設定、直接把模型當成「沒有量化」載入——不會報錯也不會印明顯警告，
# model.hf_quantizer 直接是 None，权重仍保持磁碟上原始的 float8_e4m3fn dtype、
# 數值也還是量化後的原始大小（沒有乘上 scale）。merge_and_unload() 再把 LoRA 的微小
# delta 加到這種數量級錯誤幾百倍的「假 bf16、真 fp8」權重上，存出來的 merged 模型
# 權重整組是壞的（float8_e4m3fn dtype 沒變，數值也沒被正確 dequantize），推論時會
# 產生退化的重複 token 輸出（實測合併 Mistral-Small-3.2-24B-Instruct-2506 checkpoint-7800
# 時，vLLM serve 起得來，但一問就整段輸出同一個控制字元 \x17 直到 max_tokens）。
# 修法：把底模型 config.json 自己宣告的 activation_scheme / weight_block_size /
# modules_to_not_convert 原樣帶進 FineGrainedFP8Config，只覆寫 dequantize=True，
# 讓它跟磁碟上實際的量化格式完全對上，這樣 transformers 才會真的觸發 dequantize
# 流程（把 float8_e4m3fn 權重乘上 weight_scale_inv 還原成 bf16，並清掉
# weight_scale_inv 這個殘留 buffer）。
model_kwargs = dict(low_cpu_mem_usage=True)
base_config = AutoConfig.from_pretrained(args.model_dir)
base_quant_config = getattr(base_config, "quantization_config", None)
if base_quant_config is not None:
    model_kwargs["quantization_config"] = FineGrainedFP8Config(
        activation_scheme=base_quant_config.get("activation_scheme", "dynamic"),
        weight_block_size=base_quant_config.get("weight_block_size"),
        modules_to_not_convert=base_quant_config.get("modules_to_not_convert"),
        dequantize=True,
    )
model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **model_kwargs)
model.generation_config.do_sample = True

if base_quant_config is not None:
    # 保險檢查：真的 dequantize 成功的話，底模型不該再留下任何 weight_scale_inv /
    # input_scale 這類量化中繼 buffer；如果還在，代表上面的設定又跟某個未來的底模型
    # 對不上，寧可直接中止，也不要把壞掉的權重存成看似正常的 merged 目錄。
    leftover_scale_keys = [
        k for k in model.state_dict() if k.endswith("weight_scale_inv") or k.endswith("input_scale")
    ]
    if leftover_scale_keys:
        raise RuntimeError(
            f"底模型宣告了 quantization_config，但 dequantize 後仍殘留 "
            f"{len(leftover_scale_keys)} 個量化 scale buffer（例如 {leftover_scale_keys[0]}），"
            "代表 dequantize 沒有真的生效，權重可能還是量化前的錯誤數值，已中止避免存出壞掉的模型。"
        )

print(f"[merge] 載入 LoRA adapter：{args.checkpoint}")
peft_model = PeftModel.from_pretrained(model, args.checkpoint)

print("[merge] 合併 adapter 權重回底模型（merge_and_unload）")
merged_model = peft_model.merge_and_unload()

print(f"[merge] 儲存合併後模型至：{args.merged_path}")

if base_quant_config is not None:
    # 2026-09-10 改法：底模型原本是 FP8 checkpoint 時，直接讓 merged_model.save_pretrained()
    # 存檔的話，存檔當下會透過這個 transformers 版本 core_model_loading 幫每個
    # WeightConverter 記的 reverse_op（見 transformers/integrations/finegrained_fp8.py 的
    # Fp8Dequantize.reverse_op）把上面已經 dequantize 好的 bf16 權重「還原」回
    # FP8 + weight_scale_inv 再寫檔——這是特意設計給「load 時 dequantize 只是為了用更高精度
    # 運算，存檔仍要保留原始 FP8 checkpoint 格式」的情境（round-trip），不是我們要的「永久
    # dequantize 成獨立 bf16 模型」。舊版做法是先讓 save_pretrained() 正常存檔（key 命名
    # 正確，但權重數值被還原回 FP8），存檔後再整個讀回磁碟上的 safetensors 修正一次、寫回，
    # 對 ~16GB 的資料等於多做一輪「寫錯的版本→整個讀回來→修正→整個重寫」（實測讀取+重寫
    # 約 4-5 分鐘）。
    #
    # 這裡改成跟 fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 第 9 節一致的做法：
    # 自己先呼叫「save_pretrained() 內部實際呼叫的同一個函式」revert_weight_conversion()
    # （見 transformers/modeling_utils.py 的 `state_dict = revert_weight_conversion(
    # model_to_save, state_dict)`），一次拿到「key 命名正確、但被還原回 FP8」的 state_dict，
    # 直接在記憶體裡修正（乘回 weight_scale_inv、丟掉 scale key），再自己動手分片存檔
    # 一次——全程只需要真正寫對的版本一次，不用再讀回磁碟修正。
    #
    # 風險：revert_weight_conversion 是 transformers 的內部函式（core_model_loading 模組，
    # 不是穩定公開 API），未來版本可能改名或改變行為。萬一真的失效，下面的 RuntimeError 保險
    # 檢查會直接中止、不會悄悄存出壞掉的模型——失敗模式是「跑不動」，不是「跑出錯的東西」。
    print("[merge] 用 revert_weight_conversion() 取得正確命名的 state_dict，"
          "在記憶體裡修正 FP8 round-trip 後一次存檔（不寫錯誤版本、不用讀回磁碟修正）")

    import json as _json

    import torch as _torch
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file as _save_file
    from transformers.core_model_loading import revert_weight_conversion

    _reverted_state_dict = revert_weight_conversion(merged_model, merged_model.state_dict())

    _scales = {k: v for k, v in _reverted_state_dict.items() if k.endswith("weight_scale_inv")}
    _final_state_dict = {}
    _dequantized_count = 0
    for key, tensor in _reverted_state_dict.items():
        if key.endswith("weight_scale_inv") or key.endswith("input_scale"):
            continue  # 量化中繼 key，不寫回
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
            scale = _scales.get(scale_key)
            if scale is not None:
                tensor = (tensor.float() * scale.float()).to(_torch.bfloat16)
                _dequantized_count += 1
        _final_state_dict[key] = tensor.contiguous()

    print(f"[merge] 已在記憶體裡還原 {_dequantized_count} 個權重張量成 bf16"
          "（來源 FP8 + weight_scale_inv），準備存檔")

    # 保險檢查：確認記憶體裡的最終 dict 真的沒有殘留量化 dtype 或量化中繼 key，再動手存檔。
    _remaining_bad = [
        k for k, v in _final_state_dict.items()
        if k.endswith("weight_scale_inv") or k.endswith("input_scale")
        or v.dtype not in (_torch.bfloat16, _torch.float16, _torch.float32)
    ]
    if _remaining_bad:
        raise RuntimeError(
            f"修正 FP8 round-trip 後，記憶體裡仍有 {len(_remaining_bad)} 個異常 tensor/key"
            f"（例如 {_remaining_bad[0]}），已中止，不會存出壞掉的 merged 模型。"
        )

    os.makedirs(args.merged_path, exist_ok=True)
    merged_model.config.save_pretrained(args.merged_path)
    merged_model.generation_config.save_pretrained(args.merged_path)

    _split = split_torch_state_dict_into_shards(_final_state_dict, max_shard_size="5GB")
    for _shard_file, _tensor_names in _split.filename_to_tensors.items():
        _shard = {name: _final_state_dict[name] for name in _tensor_names}
        _save_file(_shard, os.path.join(args.merged_path, _shard_file), metadata={"format": "pt"})
    if _split.is_sharded:
        _index = {"metadata": _split.metadata, "weight_map": _split.tensor_to_filename}
        with open(os.path.join(args.merged_path, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            _json.dump(_index, f, indent=2)
else:
    merged_model.save_pretrained(args.merged_path, safe_serialization=True, max_shard_size="5GB")

# 複製 base tokenizer/config 到 merged_path（底模型目錄缺 processor 專屬設定檔）。
# 刻意不複製 config.json：上面 merged_model.save_pretrained() 已經為合併後的模型寫出正確的
# config.json（若底模型原本是量化 checkpoint，合併後權重是純 bf16，config 已經沒有
# quantization_config 這個屬性）。若把 config.json 也列進來複製，會用 args.model_dir 那份
# （量化底模型的話，仍帶著原始 quantization_config）舊 config.json 蓋掉剛寫出的正確版本，導致
# 之後重新載入 merged_path 時 transformers 誤判權重的量化狀態（Ministral 實測見
# fine_tuning_vlm_trl_traffic_tune_ministral.py 的 commit 6c5da7c）。
# chat_template.json / chat_template.jinja 兩者都列進清單：不同模型家族只會有其中一種（Mistral
# 系列本地目錄通常只有 .jinja），下面的 if os.path.exists(src) 逐一檢查，缺的那個會印警告並略過。
#
# 2026-10-07：tekken.json 預設不複製。合併目錄裡已經有下面 processor.save_pretrained() 存出的
# HF tokenizer.json；兩者同時存在時，transformers 的 AutoTokenizer（vLLM 不加
# --tokenizer_mode mistral 時就是用它）會優先選 MistralCommonBackend，把 "[IMG]" 當一般文字
# 切成 [1091, 125250, 1093] 而不是 image token 10，vLLM serve 啟動即
# `Mismatch in image token count between text and input_ids`（Mistral-Small run3 實測）。
# 要用 --tokenizer_mode mistral 部署時才加 --copy_tekken。
for fname in [
    *(["tekken.json"] if args.copy_tekken else []),
    "params.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.json",
    "chat_template.jinja",
]:
    src = os.path.join(args.model_dir, fname)
    if not os.path.exists(src):
        print(f"⚠️ {src} 不存在，略過複製")
        continue
    dst = os.path.join(args.merged_path, fname)
    if fname == "params.json" and base_quant_config is not None:
        # params.json 是 vLLM `--config_format mistral` 讀的 Mistral 原生設定檔，跟
        # config.json 一樣會內嵌量化宣告（這裡是 "quantization": {"qformat_weight":
        # "fp8_e4m3", ...}），但合併後的權重已經是上面 dequantize 過的 bf16，不再是
        # FP8。若原樣複製，vLLM 讀到這個殘留的 quantization 區塊會誤判成 FP8 checkpoint
        # 去找對應的 scale tensor，但權重裡早就沒有了，會直接載入失敗或算出錯誤結果。
        # 這裡讀進來、拿掉 "quantization" 這個 key 再存，其餘欄位（維度、rope 等）不變。
        import json

        with open(src, encoding="utf-8") as f:
            params = json.load(f)
        if params.pop("quantization", None) is not None:
            print(f"[merge] {fname} 移除殘留的量化宣告（合併後權重已是 bf16）")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
    else:
        shutil.copy(src, dst)

print(f"[merge] 載入並儲存 processor（來源 checkpoint：{args.checkpoint}）")
processor = AutoProcessor.from_pretrained(args.checkpoint)
# 2026-10-07：訓練腳本把 image_processor.size.longest_edge 調成 --image_longest_edge（預設
# 1000），checkpoint 存的 processor 也帶著這個值。vLLM 啟動時用 vision_config.image_size
# （Mistral-Small 為 1540）大小的假圖預先算 image token 數，HF processor 卻把它縮到 1000，
# 兩邊 token 數對不上，serve 直接 `Expected there to be 1 prompt placeholders ... found 0`
# （run3 實測）。這裡改回底模型的 image_size。碰撞構圖都是 1000x1000，兩種設定都不會觸發
# 縮圖，前處理結果跟訓練時一致；但大於 1000 的圖推論時不會再縮到 1000，跟訓練時不同。
# image_processor.size 在 transformers 5 是 SizeDict（不是 dict），用屬性讀取。
_vision_image_size = getattr(getattr(base_config, "vision_config", None), "image_size", None)
_longest_edge = getattr(getattr(processor.image_processor, "size", None), "longest_edge", None)
if _vision_image_size and _longest_edge and _longest_edge != _vision_image_size:
    print(f"[merge] processor longest_edge {_longest_edge} -> {_vision_image_size}"
          "（對齊 vision_config.image_size，vLLM 才能正確預估 image token 數）")
    processor.image_processor.size = {"longest_edge": _vision_image_size}
processor.save_pretrained(args.merged_path)

print(f"[merge] 完成，merged 模型位於：{args.merged_path}")
