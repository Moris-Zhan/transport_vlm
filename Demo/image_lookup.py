"""以知覺雜湊 (pHash) 比對上傳圖片與 test.jsonl 中的歷史資料，取出過去的 inference 結果。"""

import io
import json
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import imagehash
from PIL import Image

DEFAULT_THRESHOLD = 5


class MatchResult(NamedTuple):
    record: dict
    distance: int


def build_index(jsonl_path: str) -> List[Tuple[dict, imagehash.ImageHash]]:
    """讀取 jsonl_path，為每筆紀錄對應的圖片計算 pHash。圖片路徑相對於 jsonl 檔案所在目錄。"""
    base_dir = Path(jsonl_path).parent
    index = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = base_dir / record["image"]
            with Image.open(image_path) as img:
                phash = imagehash.phash(img)
            index.append((record, phash))
    return index


def find_best_match(
    image_bytes: bytes,
    index: List[Tuple[dict, imagehash.ImageHash]],
    threshold: int = DEFAULT_THRESHOLD,
) -> Optional[MatchResult]:
    """回傳雜湊距離最小且落在門檻內的比對結果；找不到符合門檻的紀錄則回傳 None。"""
    with Image.open(io.BytesIO(image_bytes)) as img:
        query_hash = imagehash.phash(img)

    best_record, best_distance = None, None
    for record, phash in index:
        distance = query_hash - phash
        if best_distance is None or distance < best_distance:
            best_record, best_distance = record, distance

    if best_record is None or best_distance > threshold:
        return None
    return MatchResult(record=best_record, distance=best_distance)
