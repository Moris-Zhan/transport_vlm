"""`CollisionDiagramTuneV2` 的前一個迭代版本（`make_data_v1.py` 使用）。

與 `CollisionDiagramTuneV2` 唯一的行為差異：這裡會做跨碰撞型態去重複（同一張圖不會出現
同一種主要碰撞型態兩次），`generate_tune_v2` 之後拿掉了這個限制以增加資料多樣性。
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


class CollisionDiagramTuneV1(CollisionDiagramBase):
    """對應 `generate_tune_v1()`：根目錄 `make_data_v1.py` 使用的版本。"""

    def generate_tune_v1(self, collision_num: int, save_path: str = "", debug=False):
        # 初始化：來源、碰撞型態等
        self.get_source()  # self.source = "運研所" | "阡陌"
        self.set_collision_types(collision_num)
        self.over_overlap = 0  # 記錄重疊次數
        data = []             # 儲存畫圖的資料
        locs = []             # 已用過的位置
        check_pos = []        # 用於檢查圖例重疊的包圍盒集合
        skip_coll = []        # 被略過的碰撞，通常因重疊
        record_main_collisions = []  # 已出現的主要碰撞型態
        text = ""             # 最終問答描述
        ix = 1                # 圖例序號

        ocr_nums = 0          # 阡陌資料集：需要標記幾個傷亡數
        ocr_labels = []       # 所有傷亡數的列表

        for collision in self.collisions:
            # 取得碰撞發生地點
            loc = self.get_collision_locations(collision)
            # 取得碰撞型態標籤
            collision_label = self.get_thin_collision(loc, collision)
            # 取主要碰撞型態（"A_B" => "A"）
            main_collision = collision_label.split("_")[0]
            # 若主要碰撞型態已出現過，則略過（跨型態去重）
            if main_collision in record_main_collisions:
                continue
            record_main_collisions.append(main_collision)

            # 只保留每個地點一筆
            if loc in locs:
                continue
            locs.append(loc)

            # 轉地點名稱為座標
            x, y = self.location_to_position(loc)
            if x is None and y is None:
                continue

            # 選取該碰撞及地點相符的圖示
            icon_image = self.select_icon_image(collision, loc, collision_label)
            # 取得圖示旋轉角度
            rot_angle = self.get_los_parse_angle(main_collision, loc, icon_image)
            # 根據來源決定縮放比例
            shrink_divisor = FINAL_SHRINK_NUMBER_ICON if self.source == "阡陌" else FINAL_SHRINK_TRANSPORT_ICON
            # 根據座標決定應放置的位置，curr_box：當前圖示包圍盒
            pos, curr_box = self.place_icon(icon_image, x, y, offset_range=15, shrink_divisor=shrink_divisor)

            # 檢查新圖例是否與以往任何一個重疊
            overlap = False
            if len(check_pos) > 0:
                ious = compute_iou_vectorized(curr_box, check_pos)
                overlap_indices = np.where(ious > 0.2)[0]
                if len(overlap_indices) > 0:
                    overlap = True

            if not overlap:
                # 沒有重疊，加到已放圖例清單
                check_pos.append(curr_box)
                # 產生顯示於下方的文字說明
                way_direction, orign_description = self.get_text_way_direction(collision_label, rot_angle)

                if self.source == "阡陌":
                    # 阡陌資料集，部分才顯示傷亡標籤
                    if ocr_nums < int(collision_num / 2):
                        hurt_num, hurt_info = self.get_random_hurt()
                        ocr_nums = ocr_nums + 1
                        ocr_labels.append(hurt_num)
                    else:
                        hurt_num, hurt_info = self.get_random_hurt(force_0=True)

                    # 拼接描述文字
                    single_description = f"{hurt_info}{way_direction}，兩車在{loc}發生{main_collision}\n"
                    # 注意：overlap 這裡被重新賦值，從「是否位置重疊」的布林值，變成
                    # 「連續事件數」（1 或 6~9），交給支援 overlap 疊圖的 render 函式使用。
                    overlap, single_description = self.select_overlap(single_description)
                    # 收集所有該圖例需要的資訊（icon 位置、傷亡、描述....）
                    data.append((
                        overlap, icon_image, (0, 0), pos, rot_angle, hurt_num, inverse_location_mapping[loc],
                        main_collision, collision_label, way_direction, orign_description,
                    ))
                elif self.source == "運研所":
                    # 運研所格式組合
                    single_description = f"{way_direction}，兩車在{location_mapping[loc]}發生{main_collision}\n"
                    overlap, single_description = self.select_overlap(single_description)
                    data.append((overlap, icon_image, pos, rot_angle, location_mapping[loc], main_collision))

                # 給這一個事件的完整敘述加序號
                text += f"{ix}.{single_description}"
                ix += 1
            else:
                # 被重疊略過
                skip_coll.append((loc, main_collision))

        print(f"底圖: {self.base_image}，碰撞數量: {ix - 1}，去除重疊圖例: {len(skip_coll)}\n")
        if data:
            # 合成最後的影像
            if self.source == "阡陌":
                image = add_number_icon_to_image(self.base_image, data, False)
            else:
                image = add_transport_icon_to_image(self.base_image, data)

            # 若有指定要存結果
            if save_path:
                uid = f"{datetime.now().timestamp() * 10 ** 6:.0f}"
                output_path = f"{os.path.dirname(save_path)}/Image_{uid}.png"
                image.save(output_path, "PNG")

                # 替換其他碰撞型態的文字，讓描述符合現有邏輯
                text = text.replace("其他", "其他碰撞型態(撞固定物/失控肇事)")
                # 組裝要存的 jsonl 結構
                new_data = {
                    "source": self.source,
                    "background": self.base_image,
                    "image": output_path,
                    "text": text,
                    "icons": ix - 1,
                    "road_type": self.road_type,
                    "ocr_nums": ocr_nums,
                    "ocr_labels": ocr_labels,
                }
                # 寫入 jsonl
                save_new_data_as_jsonl(new_data, save_path)
        return ix - 1
