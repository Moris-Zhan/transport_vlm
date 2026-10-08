"""合成資料生成的逐 split 迴圈：呼叫 generator，單筆失敗時記錄後跳過、繼續下一筆，並在最後
回報成功/失敗統計。取代原本 make_data.py 裡沒有任何錯誤處理、單筆失敗就讓整個 run 中斷的
生成迴圈。
"""

import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from utils.ds_utils.assets import IconAssetNotFoundError
from utils.utils import prepare_directory

# 失敗率超過這個比例時，額外印出警告（不影響是否繼續，只是提醒使用者注意）。
FAILURE_RATE_WARNING_THRESHOLD = 0.10

# 單一 split 最多嘗試次數 = count * MAX_ATTEMPTS_MULTIPLIER；避免圖示資料本身有系統性問題
# （例如整個來源資料夾被誤刪）時，while 迴圈永遠達不到 count 而卡住。
MAX_ATTEMPTS_MULTIPLIER = 20


@dataclass
class SplitStats:
    """單一 split（train/val/test）的合成結果統計。"""

    split: str
    successes: int = 0
    failures: int = 0
    collision_nums: list = field(default_factory=list)
    diagram_times: list = field(default_factory=list)
    failure_records: list = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        total = self.successes + self.failures
        return self.failures / total if total else 0.0


def _generate_one_sample(generator_cls, generate_method_name, rand_num_min, rand_num_max, save_path):
    c = generator_cls()
    c.select_map()
    c.set_background()
    rand_num = random.randint(rand_num_min, rand_num_max)
    collision_num = getattr(c, generate_method_name)(rand_num, save_path)
    if collision_num is None:
        # generate_v1/generate_v2/generate_v3/generate_mix 這幾個版本沒有回傳實際合成數量，
        # 沿用重構前 make_data.py 的行為，以 rand_num（本次要求合成的數量）代替。
        collision_num = rand_num
    return collision_num


def _write_failure_log(data_path: str, split: str, failure_records: list) -> None:
    if not failure_records:
        return
    with open(f"{data_path}/{split}/failures.jsonl", "w", encoding="utf-8") as f:
        for record in failure_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_split_summary(stats: SplitStats) -> None:
    if stats.collision_nums:
        avg_collisions = sum(stats.collision_nums) / len(stats.collision_nums)
        avg_time = sum(stats.diagram_times) / len(stats.diagram_times)
        print(f"{stats.split}資料集，平均合成圖例數: {avg_collisions:.2f}")
        print(f"{stats.split}資料集，平均花費時間: {avg_time:.2f} s")
    print(
        f"{stats.split}資料集，成功 {stats.successes} / 失敗 {stats.failures}"
        f"（失敗率 {stats.failure_rate:.1%}）"
    )
    if stats.failure_rate > FAILURE_RATE_WARNING_THRESHOLD:
        print(
            f"⚠️ {stats.split} 失敗率 {stats.failure_rate:.1%} 超過門檻 "
            f"{FAILURE_RATE_WARNING_THRESHOLD:.0%}，請檢查 "
            f"{stats.split}/failures.jsonl 確認是否有系統性問題"
        )


def run_split(
    split: str,
    count: int,
    generator_cls,
    generate_method_name: str,
    data_path: str,
    rand_num_min: int,
    rand_num_max: int,
    max_attempts: Optional[int] = None,
) -> SplitStats:
    """合成一個 split 的 count 筆樣本。

    單筆樣本合成時若遇到 IconAssetNotFoundError（查無對應圖示檔），記錄原因後跳過、繼續
    合成下一筆，直到成功筆數達到 count；若總嘗試次數超過 max_attempts（預設
    count * MAX_ATTEMPTS_MULTIPLIER）仍未達到 count，視為系統性問題，印出目前為止的統計後
    拋出 RuntimeError（而不是無限迴圈卡住）。非 IconAssetNotFoundError 的例外一律往外拋出、
    中止執行。
    """
    if max_attempts is None:
        max_attempts = count * MAX_ATTEMPTS_MULTIPLIER

    prepare_directory(f"{data_path}/{split}")
    save_path = f"{data_path}/{split}/{split}.jsonl"
    stats = SplitStats(split=split)

    pbar = tqdm(total=count, desc=f"合成 {split}")
    attempts = 0
    try:
        while stats.successes < count:
            if attempts >= max_attempts:
                _write_failure_log(data_path, split, stats.failure_records)
                _print_split_summary(stats)
                raise RuntimeError(
                    f"{split}：嘗試 {attempts} 次仍只成功 {stats.successes}/{count} 筆，"
                    f"已超過上限 {max_attempts} 次，可能是圖示資料有系統性問題，中止執行。"
                )
            attempts += 1
            st = time.time()
            try:
                collision_num = _generate_one_sample(
                    generator_cls, generate_method_name, rand_num_min, rand_num_max, save_path
                )
            except IconAssetNotFoundError as e:
                stats.failures += 1
                stats.failure_records.append({"split": split, "attempt": attempts, "error": str(e)})
                continue
            stats.successes += 1
            stats.collision_nums.append(collision_num)
            stats.diagram_times.append(time.time() - st)
            pbar.update(1)
    finally:
        pbar.close()

    _write_failure_log(data_path, split, stats.failure_records)
    _print_split_summary(stats)
    return stats


def run(
    generator_cls,
    generate_method_name: str,
    set_num: dict,
    data_path: str,
    rand_num_min: int,
    rand_num_max: int,
) -> dict:
    """對 set_num（例如 {"train": 30000, "val": 100, "test": 100}）裡的每個 split 依序呼叫
    run_split()，回傳 {split: SplitStats}。"""
    return {
        split: run_split(split, count, generator_cls, generate_method_name, data_path, rand_num_min, rand_num_max)
        for split, count in set_num.items()
    }
