"""目前主力使用的資料格式版本（`make_data.py`）：向量化重疊檢測 + 連續事件數描述。

相較 `CollisionDiagramTuneV1`，這個版本**不會**跨碰撞型態去重複（同一張圖裡可以出現同一種
主要碰撞型態多次），這是刻意的資料多樣性設計差異，不是遺漏。
"""

from datetime import datetime
import os

import numpy as np

from utils.ds_utils.base import CollisionDiagramBase
from utils.ds_utils.io_utils import save_new_data_as_jsonl
from utils.ds_utils.rendering import (
    FINAL_SHRINK_NUMBER_ICON,
    FINAL_SHRINK_TRANSPORT_ICON,
    add_number_icon_to_image,
    add_transport_icon_to_image,
)
from utils.ds_utils.domain.geometry import compute_iou_vectorized, inverse_location_mapping, location_mapping


class CollisionDiagramTuneV2(CollisionDiagramBase):
    """對應目前主力使用的 `generate_tune_v2()`（重構前叫 `generate_tune()`）。"""

    def generate_tune_v2(self, collision_num: int, save_path: str = "", debug=False):
        self.get_source()  # 取得 icon 資源來源（運研所 or 阡陌）
        self.set_collision_types(collision_num)   # 隨機產生碰撞型態
        self.over_overlap = 0     # 追蹤重疊次數（未啟用，留底）
        data = []                 # 儲存所有 icon、描述等資訊
        locs = []                 # 避免同一個地點重複疊 icon
        check_pos = []            # 儲存已佔用的 bounding box，做 iou 判斷
        skip_coll = []            # 留存被跳過（重疊）的碰撞
        text = ""                 # 組成描述文字（最終 label 用）
        ix = 1                    # 事件流水號

        ocr_nums = 0              # 加入 OCR 數字的事件數
        ocr_labels = []           # 對應 OCR 的傷亡人數標籤

        for collision in self.collisions:   # 遍歷本次要生成的每個碰撞事件
            loc = self.get_collision_locations(collision)  # 根據碰撞隨機選地點
            collision_label = self.get_thin_collision(loc, collision)   # 獲得簡化碰撞標籤
            main_collision = collision_label.split("_")[0]             # 主碰撞型態

            if loc in locs:
                continue    # 避免同一位置出現多個 icon
            locs.append(loc)

            x, y = self.location_to_position(loc)  # 取得 icon 擺放座標
            if x is None and y is None:
                continue    # 找不到座標就跳過

            icon_image = self.select_icon_image(collision, loc, collision_label) # 選對應 icon
            rot_angle = self.get_los_parse_angle(main_collision, loc, icon_image) # 算 icon 旋轉角
            # 根據資料來源決定渲染縮放用的因子（號碼疊圖 vs. icon 疊圖）
            shrink_divisor = FINAL_SHRINK_NUMBER_ICON if self.source == "阡陌" else FINAL_SHRINK_TRANSPORT_ICON
            pos, curr_box = self.place_icon(icon_image, x, y, offset_range=15, shrink_divisor=shrink_divisor) 

            overlap = False  # 判斷是否與之前的 icon 重疊
            if len(check_pos) > 0:
                ious = compute_iou_vectorized(curr_box, check_pos)   # 算 IOU
                overlap_indices = np.where(ious > 0.2)[0]            # IOU > 0.2 視為重疊
                if len(overlap_indices) > 0:
                    overlap = True

            if not overlap:           # 不重疊才放到圖上
                check_pos.append(curr_box)   # 加入已使用方框
                way_direction, orign_description = self.get_text_way_direction(collision_label, rot_angle) # 取得方向描述

                if self.source == "阡陌":
                    # 阡陌需要做部分加入傷亡數量（作為數字 label 例子）
                    if ocr_nums < int(collision_num / 2):
                        hurt_num, hurt_info = self.get_random_hurt()
                        ocr_nums = ocr_nums + 1
                        ocr_labels.append(hurt_num)
                    else:
                        hurt_num, hurt_info = self.get_random_hurt(force_0=True) # 不要真正傷亡

                    single_description = f"{hurt_info}{way_direction}，兩車在{loc}發生{main_collision}\n"
                    # overlap 被 select_overlap() 重新設成事件連續數（給號碼疊圖專用）
                    overlap, single_description = self.select_overlap(single_description)
                    data.append((
                        overlap, icon_image, (0, 0), pos, rot_angle, hurt_num, inverse_location_mapping[loc],
                        main_collision, collision_label, way_direction, orign_description,
                    ))
                elif self.source == "運研所":
                    # 傳統運研所流程，不加入傷亡，只描述方向類型地點
                    single_description = f"{way_direction}，兩車在{location_mapping[loc]}發生{main_collision}\n"
                    overlap, single_description = self.select_overlap(single_description)
                    data.append((overlap, icon_image, pos, rot_angle, location_mapping[loc], main_collision))

                text += f"{ix}.{single_description}"  # 把描述文字編號拼成一段
                ix += 1
            else:
                skip_coll.append((loc, main_collision))     # 重疊的事件記錄在 skip_coll

        print(f"底圖: {self.base_image}，碰撞數量: {ix - 1}，去除重疊圖例: {len(skip_coll)}\n")
        if data:
            # 依據出處進行不同渲染（阡陌需標號碼和人數，運研所直接 icon 疊圖）
            if self.source == "阡陌":
                image = add_number_icon_to_image(self.base_image, data, False)
            else:
                image = add_transport_icon_to_image(self.base_image, data)

            if save_path:
                # 存檔案，加上一個唯一 id（用微秒為單位時間）
                uid = f"{datetime.now().timestamp() * 10 ** 6:.0f}"
                output_path = f"{os.path.dirname(save_path)}/Image_{uid}.png"
                image.save(output_path, "PNG")

                # label 文字預處理：把 generic 「其他」補充明確
                text = text.replace("其他", "其他碰撞型態(撞固定物/失控肇事)")
                new_data = {
                    "source": self.source,        # 資料來源
                    "background": self.base_image, # 底圖
                    "image": output_path,         # 圖片存檔位置
                    "text": text,                 # 組成的說明文字
                    "icons": ix - 1,              # icon 數量
                    "road_type": self.road_type,  # 路型
                    "ocr_nums": ocr_nums,         # 有 OCR 數字事件數
                    "ocr_labels": ocr_labels,     # OCR 標號 list
                }
                save_new_data_as_jsonl(new_data, save_path) # 存成 jsonl
        return ix - 1           # 回傳最終圖中 icon/事件的數量
