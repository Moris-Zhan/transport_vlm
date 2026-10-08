"""在 Ver.3 格式上加入運研所／阡陌混合來源。

歷史上曾有獨立的 `utils/make_utils/make_data_v4.py` driver 腳本使用這個版本，該資料夾已於
2026-07-23 移除（功能由 `make_data.py --generator CollisionDiagramMix` 取代）。
"""

from datetime import datetime
import os

from utils.ds_utils.base import CollisionDiagramBase
from utils.ds_utils.io_utils import save_new_data_as_jsonl
from utils.ds_utils.rendering import (
    FINAL_SHRINK_NUMBER_ICON,
    FINAL_SHRINK_TRANSPORT_ICON,
    add_number_icon_to_image,
    add_transport_icon_to_image,
)
from utils.ds_utils.domain.geometry import compute_iou, inverse_location_mapping, location_mapping


class CollisionDiagramMix(CollisionDiagramBase):
    """對應舊版 `generate_mix()`：每張圖隨機挑選運研所或阡陌其中一種圖示來源。"""

    def generate_mix(self, collision_num: int, save_path: str = "", debug=False):
        self.get_source()  # self.source = "運研所" | "阡陌"
        self.set_collision_types(collision_num)
        print(f"底圖: {self.base_image}，碰撞數量: {len(self.collisions)}\n")
        data = []
        locs = []
        check_pos = []
        record_main_collisions = []
        text = ""
        ix = 1
        ocr_nums = 0

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
            # shrink_divisor 要對應這張圖最終會用哪個 render 函式畫（見 generate_mix 尾端的分支）
            shrink_divisor = FINAL_SHRINK_NUMBER_ICON if self.source == "阡陌" else FINAL_SHRINK_TRANSPORT_ICON
            pos, curr_box = self.place_icon(icon_image, x, y, offset_range=5, shrink_divisor=shrink_divisor)

            overlap = False
            for prev_box in check_pos:
                iou = compute_iou(curr_box, prev_box)
                if iou > 0.01:
                    print(f"{loc}發生{main_collision} IOU={iou:.2f} 位置重疊")
                    overlap = True
                    break

            if not overlap:
                check_pos.append(curr_box)
                way_direction, orign_description = self.get_text_way_direction(collision_label, rot_angle)

                if self.source == "阡陌":
                    if ocr_nums < int(collision_num / 2):
                        hurt_num, hurt_info = self.get_random_hurt()
                        ocr_nums = ocr_nums + 1
                    else:
                        hurt_num, hurt_info = self.get_random_hurt(force_0=True)

                    text += f"{ix}.{hurt_info}{way_direction}，兩車在{loc}發生{main_collision}\n"
                    # overlap 固定傳 1（不疊貼），因為這個版本沒有連續事件數的概念
                    data.append((
                        1, icon_image, (0, 0), pos, rot_angle, hurt_num, inverse_location_mapping[loc],
                        main_collision, collision_label, way_direction, orign_description,
                    ))
                else:
                    text += f"{ix}.{way_direction}，兩車在{location_mapping[loc]}發生{main_collision}\n"
                    data.append((1, icon_image, pos, rot_angle, location_mapping[loc], main_collision))
                ix += 1

        if data:
            if self.source == "阡陌":
                image = add_number_icon_to_image(self.base_image, data, False)
            else:
                image = add_transport_icon_to_image(self.base_image, data)

            if save_path:
                uid = f"{datetime.now().timestamp() * 10 ** 6:.0f}"
                output_path = f"{os.path.dirname(save_path)}/Image_{uid}.png"
                image.save(output_path, "PNG")

                text = text.replace("其他", "其他碰撞型態(撞固定物/失控肇事)")
                new_data = {"source": self.source, "background": self.base_image, "image": output_path, "text": text}
                save_new_data_as_jsonl(new_data, save_path)
