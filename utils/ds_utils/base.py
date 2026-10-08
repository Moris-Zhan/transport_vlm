"""`CollisionDiagram` 系列類別共用的基底邏輯：底圖/碰撞型態選擇、座標換算、圖示挑選等。

每個 `generate*` 版本（見 `utils/ds_utils/generators/`）都繼承自 `CollisionDiagramBase`，
只需實作各自版本特有的合成流程。
"""

import random

import numpy as np
from PIL import Image

from utils.ds_utils import assets
from utils.ds_utils.domain.geometry import (
    inverse_location_mapping,
    rotation_angle_mapping,
    way_angle,
)
from utils.ds_utils.domain.qianmo_rules import (
    qianmo_collision_loc_map,
    qianmo_collisions,
    qianmo_way_dict,
    qianmo_weights,
)
from utils.ds_utils.domain.transport_rules import (
    transport_collision_loc_map,
    transport_collisions,
    transport_way_dict,
    transport_weights,
)
from utils.ds_utils.rendering import FINAL_SHRINK_ICON

sources = ["運研所", "阡陌"]


class CollisionDiagramBase:
    """碰撞圖例產生器的共用基底：負責底圖/碰撞型態選擇、座標與圖示查找等共用邏輯。"""

    def __init__(self):
        self.map = None
        self.base_image: str = None
        self.collisions: list[str] = None
        self.load_maps()

    def load_maps(self):
        """從 CSV 載入所有底圖與位置資訊（實際讀檔邏輯在 utils/ds_utils/assets.py）。"""
        self.maps = assets.load_base_maps()

    def select_map(self):
        """隨機選一張底圖。"""
        if self.map is None:
            self.map = random.choice(self.maps)

    def set_background(self):
        """設定底圖檔案路徑與路口類型（正交/斜交/其他）。"""
        self.base_image = self.map["Image"]
        self.way_mode = "八方位" if "正交" in self.base_image else "四方位"

        if "正交" in self.base_image:
            self.road_type = "正交"
        elif "斜交" in self.base_image:
            self.road_type = "斜交"
        else:
            self.road_type = "其他"

    def set_collision_types(self, collision_num: int):
        """隨機挑選碰撞型態。"""
        if self.source == "阡陌":
            self.collisions = random.choices(qianmo_collisions, weights=qianmo_weights, k=collision_num)
        else:
            self.collisions = random.choices(transport_collisions, weights=transport_weights, k=collision_num)

    def get_source(self):
        """隨機選擇構圖來源（運研所／阡陌）。"""
        self.source = random.choices(sources, k=1)[0]
        # self.source = "運研所"

    def get_collision_locations(self, collision):
        """根據碰撞型態，隨機選擇可能發生的地點。"""
        if self.source == "阡陌":
            locations = qianmo_collision_loc_map[collision]
        else:
            locations = transport_collision_loc_map[collision]
        return random.choice(locations)

    def get_rotation_mapping(self, angle: int):
        """每個固定角度位置，在旋轉後代表的方位。"""
        return {k: way_angle[(k + angle) % 360] for k in way_angle.keys()}

    def get_random_hurt(self, force_0=False):
        """隨機產生受傷人數（0~15 人），force_0 時固定為 0。"""
        num = 0 if force_0 else random.randrange(0, 15)
        # num = 0 if force_0 else random.randrange(0, 9)
        info = f"受傷人數{num}人，" if num != 0 else ""
        return num, info

    def select_overlap(self, description, max_overlap=9):
        """
        根據當前已經記錄的重疊次數（self.over_overlap），決定這一次是否要加上連續事件的描述。

        流程如下：
        1. 初始的 overlap 預設為 1，代表沒有連續事件。
        2. 如果 self.over_overlap 小於 2（表示尚未達到最大容許的疊加次數限制），
           則有機會額外加入連續事件的描述，執行一次隨機判斷：
            - 隨機從 0、1 中選擇一個作為 is_overlap。
            - 若抽中 is_overlap 為 1，則會從 [6, max_overlap] 範圍隨機選擇一個整數作為 overlap，
              表示此次為多事件疊加，並在 description 前方加入 `"連續事件數共{overlap}件，"` 的描述字串。
            - 每次嘗試都會將 self.over_overlap 計數加 1，限制最多改寫 2 次疊加情境。
        3. 回傳 overlap（事件數量，預設為 1，或為隨機疊加數）與新的 description（事件敘述）。

        參數:
            description (str): 初步生成的事件描述文字。
            max_overlap (int): 連續事件的最大疊加數，預設為 9。

        回傳:
            (int, str): overlap 事件數量, 重新描述後的 description 字串。
        """
        ################################################################
                                # 原始疊加 (舊版)
        ################################################################
        overlap = 1
        if self.over_overlap < 2:
            is_overlap = random.choice([0, 1])
            if is_overlap:
                overlap = random.choice(range(6, max_overlap + 1))
                description = f"連續事件數共{overlap}件，{description}"
            self.over_overlap = self.over_overlap + 1
        ################################################################
                                # 不疊加 (新版)
        ################################################################
        # overlap = 1
        return overlap, description
   

    def get_text_way_direction(self, collision_name, rot_angle, debug=False):
        """依旋轉角度換算出正確的車流方向文字描述。"""

        def convert_description(description: str) -> str:
            """逐字轉換，避免重複替換。"""
            return "".join(conversion_dict.get(char, char) for char in description)

        rotation_mapping = self.get_rotation_mapping(rot_angle)
        conversion_dict = {way_angle[angle]: rotation_mapping[angle] for angle in way_angle.keys()}

        way_dictionary = qianmo_way_dict if self.source == "阡陌" else transport_way_dict[self.way_mode]

        rotation_description = convert_description(way_dictionary[collision_name])
        rotation_description = rotation_description.replace("南西", "西南").replace("南東", "東南")
        rotation_description = rotation_description.replace("北西", "西北").replace("北東", "東北")

        if debug and rot_angle == 90:
            print(f"----------{rot_angle}----------------------")
            print(way_dictionary[collision_name])
            print(rotation_description)

        return rotation_description, way_dictionary[collision_name]

    def get_thin_collision(self, location, collision):
        """根據碰撞型態，和可能發生的地點，隨機選擇可能發生的細分類碰撞。"""
        if self.source == "阡陌":
            return random.choice(assets.QIANMO_THIN_COLLISIONS[collision][location])
        return assets.TRANSPORT_THIN_COLLISIONS[collision][location]

    def location_to_position(self, location):
        """將地點名稱轉換為實際座標位置。"""
        if self.source == "阡陌":
            positions = self.map.get(inverse_location_mapping[location])
        else:
            positions = self.map.get(location)

        if positions:
            return random.choice(positions)
        return None, None

    def select_icon_image(self, collision, location, sml_collision):
        """從資料夾中選擇對應的圖示檔（實際查找邏輯在 utils/ds_utils/assets.py）；查無檔案
        時讓 assets.find_icon_image() 的 IconAssetNotFoundError 往外傳，由呼叫端（
        utils/ds_utils/pipeline.py）決定跳過並記錄。"""
        return assets.find_icon_image(self.source, collision, location, sml_collision)

    def get_los_parse_angle(self, collision, location, icon_image):
        if self.source == "阡陌":
            return rotation_angle_mapping[inverse_location_mapping[location]]
        return rotation_angle_mapping[location]

    def place_icon(self, icon_image_path: str, x: int, y: int, offset_range: int = 15,
                    shrink_divisor: float = FINAL_SHRINK_ICON):
        """算出圖示縮小後，實際要貼到底圖上的座標與外框（供重疊檢測使用）。

        shrink_divisor 必須跟實際負責畫這個圖示的 render 函式（utils/ds_utils/rendering.py）
        最終的縮放倍率一致，呼叫端要對應傳入：
            - add_icon_to_image()            -> FINAL_SHRINK_ICON
            - add_number_icon_to_image()      -> FINAL_SHRINK_NUMBER_ICON
            - add_transport_icon_to_image()   -> FINAL_SHRINK_TRANSPORT_ICON

        2026-07-24 修正：舊版做法會先把圖示強制縮成 200x200 再除以 4 來估計外框，這個
        「先縮到 200x200」的步驟不對應任何一個 render 函式的實際邏輯（三個 render 函式都是
        直接用原始解析度 + 各自的縮放倍率），導致估計出來的外框只有實際畫面的 1/9~1/16
        大——重疊偵測形同虛設，圖示視覺上明顯疊在一起卻會被判定沒有重疊。

        現在直接用原始解析度計算 alpha bbox，套用跟 render 函式一致的縮放倍率；圖示旋轉角度
        不一定是 90 度的倍數，這裡用縮放後寬高的較大值當正方形邊長來保守估計外框，避免因為
        沒精確模擬旋轉後的外框形狀而低估。
        """
        pil_image = Image.open(icon_image_path)

        alpha = pil_image.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            content_w, content_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            content_w, content_h = pil_image.size

        content_w = content_w / shrink_divisor
        content_h = content_h / shrink_divisor
        side = max(1, int(max(content_w, content_h)))

        x_offset = random.randint(-offset_range, offset_range)
        y_offset = random.randint(-offset_range, offset_range)

        pos = (
            np.clip(x + x_offset - side // 2, 0, 1000 - side),
            np.clip(y + y_offset - side // 2, 0, 1000 - side),
        )
        curr_box = (pos[0], pos[1], pos[0] + side, pos[1] + side)
        return pos, curr_box
