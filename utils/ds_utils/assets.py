"""底圖與碰撞圖示等靜態素材的載入與查找（會做檔案系統 I/O，跟 domain/ 的純函式分開）。"""

import os
import random
from glob import glob

import pandas as pd

from utils.ds_utils.domain.geometry import location_csv, locations
from utils.ds_utils.domain.qianmo_rules import qianmo_collisions
from utils.ds_utils.domain.transport_rules import transport_collision_loc_map, transport_collisions

BASE_MAPS_DIR = "images/base_maps"
TRANSPORT_ICON_DIR = "images/collision_icons(運研所)"
QIANMO_ICON_DIR = "images/collision_icons(阡陌)"


class IconAssetNotFoundError(Exception):
    """指定的碰撞型態/地點/細分類組合在圖示資料夾裡找不到對應檔案。

    這是 make_data.py 合成迴圈裡「單筆失敗時跳過並繼續」機制要捕捉的例外類型；取代原本
    `random.choice([])` 在同樣情況下丟出、看不出原因的裸 IndexError。
    """

    def __init__(self, source: str, collision: str, location: str, sml_collision: str, pattern: str):
        self.source = source
        self.collision = collision
        self.location = location
        self.sml_collision = sml_collision
        self.pattern = pattern
        super().__init__(
            f"找不到圖示檔：source={source}, collision={collision}, "
            f"location={location}, sml_collision={sml_collision}（搜尋樣式：{pattern}）"
        )


def load_base_maps() -> list:
    """從 CSV 載入所有底圖與位置資訊，回傳 list[dict]（每筆是一張底圖的資訊）。"""
    df_base_maps = pd.read_csv(f"{BASE_MAPS_DIR}/{location_csv}").fillna("[]")
    df_base_maps[locations] = df_base_maps[locations].map(eval)
    return df_base_maps.to_dict(orient="records")


def find_icon_image(source: str, collision: str, location: str, sml_collision: str) -> str:
    """依來源/碰撞型態/地點/細分類挑一張圖示檔路徑；查無檔案時丟出 IconAssetNotFoundError。"""
    if source == "阡陌":
        pattern = f"{QIANMO_ICON_DIR}/{collision}/{location}/{sml_collision}*"
    else:
        pattern = f"{TRANSPORT_ICON_DIR}/{collision}/{sml_collision}.png"

    icon_images = glob(pattern)
    if not icon_images:
        raise IconAssetNotFoundError(source, collision, location, sml_collision, pattern)
    return random.choice(icon_images)


def _build_transport_thin_collisions() -> dict:
    """依類型與地點列出運研所圖示的細分類名稱（main_col -> region -> small_col）。"""
    return {
        main_col:
            {region: small_col.replace(".png", "")
             for region in transport_collision_loc_map[main_col]
             for small_col in os.listdir(os.path.join(TRANSPORT_ICON_DIR, main_col))
             }
        for main_col in transport_collisions
    }


def _build_qianmo_thin_collisions() -> dict:
    """依類型與地點列出阡陌圖示的細分類名稱（col -> loc -> [small_col, ...]）。"""
    return {
        col: {
            loc: [os.path.basename(p).split(".")[0] for p in glob(f"{QIANMO_ICON_DIR}/{col}/{loc}/*.png")]
            for loc in os.listdir(f"{QIANMO_ICON_DIR}/{col}")
            if os.path.isdir(os.path.join(f"{QIANMO_ICON_DIR}/{col}", loc))
        }
        for col in qianmo_collisions
    }


# 跟原本 utils/transport_utils.py、utils/qianmo_utils.py 一樣，在模組載入時就掃描一次
# images/collision_icons(...) 底下的資料夾/檔案；base.py 的 get_thin_collision() 會查這兩個表。
TRANSPORT_THIN_COLLISIONS = _build_transport_thin_collisions()
QIANMO_THIN_COLLISIONS = _build_qianmo_thin_collisions()
