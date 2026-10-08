"""合成資料的輸出與驗證工具。"""

import json
import os
from typing import Optional, Tuple

from PIL import Image as PILImage


def save_new_data_as_jsonl(new_data: dict, filename: str) -> None:
    """將一筆資料附加寫入 JSONL 檔（每行一筆 JSON）。"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(new_data, ensure_ascii=False) + "\n")


def check_image_record(record: dict) -> Tuple[bool, Optional[str]]:
    """檢查單筆紀錄的 image 欄位是否存在且能正常開啟。

    回傳 (是否正常, 問題說明)：正常時問題說明為 None；圖片路徑缺失/檔案不存在時為
    "missing"；檔案存在但無法開啟（損毀）時為對應的例外訊息。
    """
    image_path = record.get("image", "")
    if not image_path or not os.path.exists(image_path):
        return False, "missing"
    try:
        with PILImage.open(image_path) as img:
            img.verify()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, None


def is_valid_record(record: dict) -> bool:
    """給 make_data.py 的 build_dataset()（dataset.filter()）使用：background 非空、且 image
    存在並能正常開啟才算可用。與 check_dataset() 共用 check_image_record()，避免同一份驗證
    邏輯被實作兩次而彼此漂移。
    """
    if not record.get("background"):
        return False
    ok, _ = check_image_record(record)
    return ok
