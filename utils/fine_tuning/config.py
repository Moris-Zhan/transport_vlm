"""微調的可設定項：LoRA 超參數與訓練設定的 dataclass，以及對應的 CLI 參數解析。"""

import argparse
from dataclasses import dataclass, fields


@dataclass
class TrainingConfig:
    """訓練的完整可設定項。每個欄位的預設值都沿用重構前腳本的行為（原本是 CLI 參數預設值
    的欄位、或原本寫死在 LoRA/SFTConfig 裡的值）。"""

    # 原本就是 CLI 參數的欄位
    model_id: str = "Mistral-Small-3.2-24B-Instruct-2506"
    num_train_epochs: float = 5
    max_steps: int = 0
    learning_rate: float = 2e-4
    data_root: str = "data/collision_diagram/synthetic_tune_v2"
    resource_mode: str = "auto"
    # 2026-09-08 調整：batch_size/accumulation_steps 原本是 1/16，改成 4/4。
    # 1 * 16 == 4 * 4 == 16，effective batch size（per-device batch size × gradient
    # accumulation steps）不變，只是重新分配兩者比例：per-device batch size 提高、
    # accumulation steps 降低，用更多單步 GPU 記憶體換取更少的累積步數。
    batch_size: int = 8
    accumulation_steps: int = 2
    num_workers: int = 8
    new_run: bool = False
    # 2026-09-10 新增：訓練完是否跳過腳本內建的「合併 LoRA 回底模型」那一節（目前只有
    # fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 會讀這個欄位並據此跳過合併，
    # 其餘腳本忽略）。給 smoke_test_train_merge_vllm.sh 這類外部流程用：訓練完只存 LoRA
    # adapter 就結束，合併改交給獨立的 scripts/merge_lora_checkpoint.py 執行，避免兩邊各自
    # merge 一次、重複讀寫整份底模型權重。
    skip_merge: bool = False

    # 原本寫死在腳本裡、這次改成可設定的欄位
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_r: int = 16
    # image_longest_edge: int = 672
    image_longest_edge: int = 1000
    target_effective_batch_size: int = 16
    logging_steps: int = 100
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "linear"

    # 2026-09-03 新增：原本沒有這個欄位（等同 HF Trainer 預設值 None，永不清除舊
    # checkpoint），每 save_steps（HF 預設 500）就多存一份 LoRA adapter checkpoint，
    # 完整訓練下來會累積十幾份、把磁碟空間吃光。save_total_limit 讓 Trainer 存新的之前
    # 自動刪最舊的，只保留最新的 N 份；沿用 get_last_checkpoint() 的續跑邏輯不受影響
    # （它一律抓「最新」那份）。
    save_total_limit: int = 2

    # 2026-09-07 新增：原本沒有這個欄位，SFTConfig 沒設定 save_steps 就一律吃 HF Trainer
    # 預設值 500。存成可設定欄位，方便正式訓練時視情況調整存檔頻率（例如這台機器已知會被
    # 硬體/驅動層級中斷，調低 save_steps 讓中斷後續跑時最多重算的 step 數變少）。
    # 2026-09-09 修正：這裡曾經被改成 10（本次修正前的值），但每份 checkpoint 實測 635MB，
    # 13 秒/step 時等於每約 130 秒就寫 635MB 並刪一份舊的，整個訓練下來持續的高頻寫入 I/O
    # 會讓訓練 process 頻繁卡在磁碟 D state、外觀近似「整機沒反應」，且很傷 SSD；下面這個
    # docstring 當時也還寫著「預設值維持 500」跟實際的 10 互相矛盾。改回 100：兩次 checkpoint
    # 間隔拉長到約 22 分鐘（仍遠比整段訓練短，SIGSEGV 中斷時最多重算 22 分鐘），寫入量降為
    # 十分之一。
    save_steps: int = 200

    # 2026-09-14 新增：原本這裡完全沒有設定 eval_strategy，SFTTrainer 雖然收到了
    # eval_dataset（三支腳本都有 eval_dataset=eval_dataset），但 HF Trainer 沒指定
    # eval_strategy 時預設整場訓練完全不會跑驗證——追加訓練後回頭想確認有沒有 overfitting，
    # 才發現 wandb log 裡從頭到尾只有 train loss/accuracy，沒有任何 eval 數字可以核對
    # （見對話記錄 2026-09-14）。預設改成 "steps"，搭配下面的 eval_steps，讓以後每次訓練
    # 都能同步看到 train/eval loss 曲線有沒有分岔。
    eval_strategy: str = "steps"
    # 預設沿用跟 save_steps 相同的頻率（見上面 save_steps 的說明：100 step ≈ 22 分鐘），
    # 沒有特別理由要跟存檔頻率不同，需要更密集/更疏的驗證頻率再另外指定。
    eval_steps: int = 200

    # 2026-09-14 新增：原本沒有這個欄位，SFTConfig 沒設定 weight_decay 就一律吃 HF Trainer
    # 預設值 0.0（無 L2 正則化）。存成可設定欄位，方便長時間/多 epoch 訓練（例如中途用
    # --num_train_epochs 加碼續跑）時視情況加上正則化抑制過擬合；預設值維持 0.0，不改變
    # 既有訓練的行為，需要時用 --weight_decay 明確指定（例如 0.01）。
    weight_decay: float = 0.0

    # 2026-09-14 新增：原本沒有 early stopping，訓練會一路跑滿 --num_train_epochs/--max_steps，
    # 即使 eval loss 早就不再下降（甚至開始上升、overfitting）也不會提早停。early_stopping_patience
    # 是「連續幾次 eval（依 --eval_steps 頻率）沒有比目前最佳 eval loss 更好」就停止訓練，預設 15
    # 次（搭配預設 eval_steps=100，約每 100 step 評估一次，等於連續 1500 step 沒進步才停）。
    # 設為 0 或負數則不啟用 early stopping（維持原本「跑滿設定長度」的行為）。
    early_stopping_patience: int = 15
    # early_stopping_threshold：eval loss 至少要下降超過這個量才算「有進步」（避免浮點雜訊等級
    # 的微小波動被算成進步、讓 patience 計數器一直被重置）。預設 0.0 沿用 HF
    # EarlyStoppingCallback 的原生預設值，只要求「更小」即可。
    early_stopping_threshold: float = 0.0003

    # 2026-10-07 新增：原本寫死在各腳本的 attn_implementation 改成可設定欄位。共用預設值維持
    # "sdpa"（PyTorch 內建、不需要另外安裝），不改變既有腳本的行為；要用 flash-attn 時傳
    # --attn_implementation flash_attention_2，需要該 python 環境裝有跟 torch 版本對應編譯的
    # flash_attn（見 scripts/setup_flash_attn_env.sh）。
    attn_implementation: str = "flash_attention_2"

    # 2026-10-07 新增：多卡訓練時放在 GPU0 的語言模型層數（其餘層平均分給其他卡）。None 表示
    # 由腳本自行決定；目前只有 ministral_simplified.py 使用。
    gpu0_layer_count: int | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    """建立 argparse parser，每個參數的預設值都取自 TrainingConfig 的 dataclass 預設值，
    避免兩邊各自寫一份預設值、日後改一邊忘了改另一邊。
    """
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="LoRA/SFT 微調 Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument(
        "--model_id",
        type=str,
        choices=[
            "Mistral-Small-3.2-24B-Instruct-2506",
            "GLM-4.1V-9B-Thinking",
            "Ministral-3-8B-Instruct-2512",
        ],
        default=defaults.model_id,
        help="要微調的模型，對應本地 models/<model_id> 目錄",
    )
    parser.add_argument("--num_train_epochs", type=float, default=defaults.num_train_epochs, help="訓練 epoch 數")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=defaults.max_steps,
        help="限制最大訓練 step 數；設為 0（預設）則不限制 step 數，改以 --num_train_epochs 決定訓練長度",
    )
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate, help="學習率（QLoRA 論文建議 2e-4）")
    parser.add_argument(
        "--data_root",
        type=str,
        default=defaults.data_root,
        help=(
            "訓練資料所在目錄，例如 data/collision_diagram/synthetic_tune_v1。"
            "其他可選版本：synthetic / synthetic_way / synthetic_way_hurt / synthetic_mix / "
            "synthetic_tune / synthetic_tune_v2"
        ),
    )
    parser.add_argument(
        "--resource_mode",
        type=str,
        choices=["auto", "manual"],
        default=defaults.resource_mode,
        help="auto: 自動偵測 GPU/CPU 資源決定 batch size 等設定；manual: 改用下面三個參數手動指定",
    )
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size, help="[manual 模式] per_device_train_batch_size")
    parser.add_argument("--accumulation_steps", type=int, default=defaults.accumulation_steps, help="[manual 模式] gradient_accumulation_steps")
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers, help="[manual 模式] dataloader_num_workers")
    parser.add_argument(
        "--new_run",
        action="store_true",
        help=(
            "開一輪全新訓練：output_dir 用「目前最大 run index + 1」重新命名，從 base model 重頭訓練。"
            "不加這個參數時，預設一律自動接續目前最大 index 的 run（讀該資料夾內既有 checkpoint 續跑）。"
        ),
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help=(
            "訓練完只存 LoRA adapter，跳過腳本內建的合併回底模型那一節（目前只有 "
            "fine_tuning_vlm_trl_traffic_tune_ministral_fastmerge.py 會讀這個參數，其餘腳本忽略）。"
            "合併改用 scripts/merge_lora_checkpoint.py 對存好的 adapter 目錄手動執行。"
        ),
    )
    parser.add_argument("--lora_alpha", type=int, default=defaults.lora_alpha, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=defaults.lora_dropout, help="LoRA dropout")
    parser.add_argument("--lora_r", type=int, default=defaults.lora_r, help="LoRA rank r")
    parser.add_argument("--image_longest_edge", type=int, default=defaults.image_longest_edge, help="圖片縮圖後最長邊像素數")
    parser.add_argument(
        "--target_effective_batch_size",
        type=int,
        default=defaults.target_effective_batch_size,
        help="auto 模式下，資源偵測要維持的目標 effective batch size",
    )
    parser.add_argument("--logging_steps", type=int, default=defaults.logging_steps, help="訓練 log 間隔 step 數")
    parser.add_argument("--warmup_ratio", type=float, default=defaults.warmup_ratio, help="warmup ratio（QLoRA 論文建議值）")
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm, help="梯度裁剪上限（QLoRA 論文建議值）")
    parser.add_argument("--lr_scheduler_type", type=str, default=defaults.lr_scheduler_type, help="learning rate scheduler 類型")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=defaults.save_total_limit,
        help=(
            "最多保留幾份 checkpoint，存新的之前自動刪最舊的（避免無限累積吃光磁碟空間）；"
            "設為 0 或負數等同 Trainer 的預設行為，不設上限、永遠不刪舊的"
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=defaults.save_steps,
        help="每幾個 step 存一次 checkpoint（HF Trainer 原生預設 500，這支腳本預設改為 100）",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default=defaults.eval_strategy,
        help=(
            "訓練中驗證集評估頻率：no 不評估（HF Trainer 原生預設）、"
            "steps 每 --eval_steps 評估一次（這支腳本預設值）、epoch 每個 epoch 結束評估一次"
        ),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=defaults.eval_steps,
        help="--eval_strategy=steps 時，每幾個 step 跑一次驗證集評估（預設沿用 --save_steps 的頻率）",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=defaults.weight_decay,
        help="AdamW weight decay（L2 正則化），預設 0.0（HF Trainer 原生預設值，不額外加正則化）",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=defaults.early_stopping_patience,
        help=(
            "連續幾次 eval（依 --eval_steps 頻率）沒有比目前最佳 eval loss 更好就停止訓練，"
            "預設 15。設為 0 或負數關閉 early stopping（跑滿 --num_train_epochs/--max_steps）。"
            "需要 --eval_strategy 不是 no（不評估就無從判斷 eval loss 有沒有進步）。"
        ),
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=defaults.early_stopping_threshold,
        help="eval loss 至少要下降超過這個量才算「有進步」（重置 patience 計數器），預設 0.0",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["sdpa", "eager", "flash_attention_2"],
        default=defaults.attn_implementation,
        help="模型的 attention 實作；flash_attention_2 需要環境裡裝有對應 torch 版本編譯的 flash_attn",
    )
    parser.add_argument(
        "--gpu0_layer_count",
        type=int,
        default=defaults.gpu0_layer_count,
        help="多卡訓練時放在 GPU0 的語言模型層數（其餘層平均分給其他卡）；不指定則由腳本決定",
    )
    return parser


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """把 argparse 解析出的 Namespace 轉成 TrainingConfig；每個 TrainingConfig 欄位都要求
    args 裡有同名屬性（由 build_arg_parser() 保證），缺一個都會直接 AttributeError，避免兩邊
    欄位悄悄對不上。
    """
    field_names = [f.name for f in fields(TrainingConfig)]
    return TrainingConfig(**{name: getattr(args, name) for name in field_names})
