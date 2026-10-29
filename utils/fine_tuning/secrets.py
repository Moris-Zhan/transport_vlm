"""讀取 configs/secrets.yaml 裡的密鑰（wandb_api_key、hf_token）。

這個檔案刻意不進 git（見 .gitignore），避免重蹈之前 wandb API key / HuggingFace token 被寫死在
程式碼裡、不小心 commit 進 git 歷史的覆轍。configs/secrets.yaml.example 是委交進 git 的空白範本，
第一次使用時複製一份出來、填入自己的 key 即可。
"""

import yaml


def _load_secret(field: str, path: str, hint: str = "") -> str:
    """從 YAML 設定檔讀取指定欄位；檔案不存在或欄位空白都直接報錯，不接受任何寫死在程式碼裡的
    預設值。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise RuntimeError(
            f"找不到 {path}，請先複製 configs/secrets.yaml.example 為 {path} 並填入 {field}{hint}"
        ) from None

    value = data.get(field)
    if not value:
        raise RuntimeError(f"{path} 裡沒有設定 {field}（欄位是空的或不存在）")
    return value


def load_wandb_api_key(path: str = "configs/secrets.yaml") -> str:
    """讀取 wandb_api_key。"""
    return _load_secret(
        "wandb_api_key",
        path,
        hint="（本機驗證/不需要真的記錄到 wandb 時，可以搭配 `export WANDB_MODE=offline` 使用，"
        "wandb_api_key 填任意字串即可，wandb 就不會嘗試連線）。",
    )


def load_hf_token(path: str = "configs/secrets.yaml") -> str:
    """讀取 hf_token（scripts/download_model.py 下載需要授權的模型時使用）。"""
    return _load_secret("hf_token", path)
