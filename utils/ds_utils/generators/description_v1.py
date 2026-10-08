"""肇事型態描述 Ver.1：只標註「在哪個地點發生什麼肇事類型」。

歷史上曾有獨立的 `utils/make_utils/make_data_v1.py` driver 腳本使用這個版本，該資料夾已於
2026-07-23 移除（功能由 `make_data.py --generator CollisionDescriptionV1` 取代）。
"""

from datetime import datetime
import os

from utils.ds_utils.base import CollisionDiagramBase
from utils.ds_utils.io_utils import save_new_data_as_jsonl
from utils.ds_utils.rendering import FINAL_SHRINK_ICON, add_icon_to_image
from utils.ds_utils.domain.geometry import compute_iou, location_mapping


class CollisionDescriptionV1(CollisionDiagramBase):
    """肇事型態描述 Ver.1：只有路段與肇事類型，不含行車方向、不含傷亡人數。"""

    def generate_v1(self, collision_num: int, save_path: str = ""):
        # 這個版本只支援運研所這套詞彙（不支援阡陌），固定設定來源，
        # 不依賴呼叫端事先設定 self.source。
        self.source = "運研所"
        self.set_collision_types(collision_num)
        print(f"底圖: {self.base_image}，碰撞數量: {len(self.collisions)}\n")
        data = []
        locs = []
        record_main_collisions = []
        text = ""
        ix = 1
        check_pos = []

        for collision in self.collisions:
            loc = self.get_collision_locations(collision)
            collision_label = self.get_thin_collision(loc, collision)
            main_collision = collision_label.split("_")[0]
            if main_collision in record_main_collisions:
                continue
            record_main_collisions.append(main_collision)

            if loc in locs:
                continue
            locs.append(loc)

            x, y = self.location_to_position(loc)
            if x is None and y is None:
                continue

            icon_image = self.select_icon_image(collision, loc, collision_label)
            rot_angle = self.get_los_parse_angle(main_collision, loc, icon_image)
            pos, curr_box = self.place_icon(icon_image, x, y, offset_range=15, shrink_divisor=FINAL_SHRINK_ICON)

            overlap = False
            for prev_box in check_pos:
                iou = compute_iou(curr_box, prev_box)
                if iou > 0.01:
                    print(f"{loc}發生{main_collision} IOU={iou:.2f} 位置重疊")
                    overlap = True
                    break

            if not overlap:
                check_pos.append(curr_box)
                text += f"{ix}.在{loc}發生{main_collision}\n"
                data.append((icon_image, pos, rot_angle, location_mapping[loc], main_collision))
                ix += 1

        if data:
            image = add_icon_to_image(self.base_image, data)

            if save_path:
                uid = f"{datetime.now().timestamp() * 10 ** 6:.0f}"
                output_path = f"{os.path.dirname(save_path)}/Image_{uid}.png"
                image.save(output_path, "PNG")

                text = text.replace("其他", "其他碰撞型態(撞固定物/失控肇事)")
                new_data = {"background": self.base_image, "image": output_path, "text": text}
                save_new_data_as_jsonl(new_data, save_path)
