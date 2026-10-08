"""把這次訓練實際生效的三份設定（TrainingConfig / SFTConfig / LoraConfig）各自存成 YAML，
放進這次訓練的 output_dir 根目錄。

動機：這支腳本會被外部監督迴圈（run_formal_finetune.sh）重複啟動，每次正式訓練事後想確認
「這個 run 到底是用哪些參數跑的」都要翻指令歷史或猜，直接存成檔案放在 output_dir 底下最直觀。
"""

import dataclasses
import enum
import os

import yaml


def _to_yaml_safe(value):
    """遞迴把不是 YAML 基本型別（str/int/float/bool/None/list/dict）的值轉成安全型別，
    避免 dataclasses.asdict() 挖出來的 Enum（例如 SFTConfig 的 IntervalStrategy、LoraConfig
    的 PeftType）讓 yaml.safe_dump 直接丟 RepresenterError。

    Enum 判斷必須放在 str/int 判斷「之前」：transformers 的 IntervalStrategy 這類 enum 是
    `class IntervalStrategy(str, Enum)`，實例同時也是 str 的子類別。isinstance(value, str)
    會先比對成立，若把基本型別判斷放前面，Enum 分支永遠不會被走到，PyYAML 拿到一個它不認得
    的 str 子類別一樣會丟 RepresenterError（實測發生過：save_strategy=IntervalStrategy.NO
    導致 sft_config.yaml 存檔直接炸掉，訓練程序因此整個中斷）。"""
    if isinstance(value, enum.Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_yaml_safe(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_yaml_safe(dataclasses.asdict(value))
    return str(value)


def dump_run_configs(output_dir: str, training_config, sft_args, peft_config) -> None:
    """training_config / sft_args / peft_config 都必須是 dataclass 實例（TrainingConfig /
    SFTConfig / LoraConfig 皆是），三者各自存成 output_dir 底下的一個 YAML 檔，每次呼叫都會
    覆寫（反映「這次啟動當下實際生效」的參數；外部監督迴圈重跑時如果帶的 CLI 參數不同，檔案
    內容也會跟著更新）。"""
    os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)

    files_and_configs = {
        "training_config.yaml": training_config,
        "sft_config.yaml": sft_args,
        "peft_config.yaml": peft_config,
    }
    for fname, config_obj in files_and_configs.items():
        safe_dict = _to_yaml_safe(dataclasses.asdict(config_obj))
        with open(os.path.join(output_dir, "configs", fname), "w", encoding="utf-8") as f:
            yaml.safe_dump(safe_dict, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print("configs saved to", os.path.join(output_dir, "configs"))
