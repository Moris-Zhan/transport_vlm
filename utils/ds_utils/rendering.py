"""把碰撞圖示合成到底圖上的繪製函式。

每個 generate* 版本會依資料格式（是否含受傷人數、是否為運研所/阡陌混合來源、是否有重疊
事件數）挑選對應的繪製函式；因此這裡維持成一組獨立函式而非類別方法。

效能優化：碰撞圖示的讀檔+解碼已經在 utils/ds_utils/image_ops.py 的 white_to_transparent()
內快取（見該檔案的 _load_rgba()）。底圖（images/base_maps/ 下固定的幾十張）在這裡另外用
_load_base_image()/_load_edge_enhanced_base_image() 快取——理由相同：跑 make_data.py 大量
生成（例如 30,000 張）時，同一個底圖檔案會被重複開啟、resize、（運研所版本）邊緣強化非常
多次。快取回傳的物件都只會被 .copy() 後才拿去 paste()，不會被原地修改，是安全的。
"""
import math

import functools
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from utils.ds_utils.image_ops import crop_transparent, white_to_transparent

 # (舊版)
font = ImageFont.truetype("Microsoft-JhengHei.ttf", size=11)

 # (新版)
big_font = ImageFont.truetype("Microsoft-JhengHei.ttf", size=25)
angle_font = ImageFont.truetype("Microsoft-JhengHei.ttf", size=15)

# 圖示縮小的門檻與比例。運研所（TRANSPORT_ICON_SHRINK_DIVISOR，再乘一次
# 1/TRANSPORT_ICON_SECOND_SHRINK_RATIO）跟阡陌/預設（ICON_SHRINK_DIVISOR）兩套圖示的縮放
# 係數不一樣，推測是因為兩邊原始圖檔解析度不同、需要不同縮放比例才會視覺對齊；程式碼裡
# 沒有寫明這個假設，之後如果要統一/調整縮放邏輯，需要先實際比對兩套圖示的原始尺寸，不要
# 直接假設數字可以互換。
ICON_RESIZE_THRESHOLD = 200  # 圖示邊長超過這個值（px）才進行縮小
ICON_SHRINK_DIVISOR = 4  # add_icon_to_image() 使用
TRANSPORT_ICON_SHRINK_DIVISOR = 2  # add_transport_icon_to_image() 第一次縮小（縮放前）
TRANSPORT_ICON_SECOND_SHRINK_RATIO = 1.5  # add_transport_icon_to_image() 第二次縮小（旋轉後）
NUMBER_ICON_SHRINK_RATIO = 1.5  # add_number_icon_to_image() 縮小（畫完受傷人數後）

# 各 render 函式「原始解析度 -> 最終貼到底圖上」的總縮放倍率，供
# CollisionDiagramBase.place_icon() 估計重疊判斷框時使用，務必跟對應的 render 函式保持一致
# （見 utils/ds_utils/base.py 的 place_icon() docstring）。
FINAL_SHRINK_ICON = ICON_SHRINK_DIVISOR  # add_icon_to_image() 對應
FINAL_SHRINK_NUMBER_ICON = ICON_SHRINK_DIVISOR * NUMBER_ICON_SHRINK_RATIO  # add_number_icon_to_image() 對應
FINAL_SHRINK_TRANSPORT_ICON = TRANSPORT_ICON_SHRINK_DIVISOR * TRANSPORT_ICON_SECOND_SHRINK_RATIO  # add_transport_icon_to_image() 對應


@functools.lru_cache(maxsize=None)
def _load_base_image(base_image_path: str, base_size: tuple) -> Image.Image:
    """讀取並縮放底圖，結果依 (路徑, 尺寸) 快取。

    回傳的物件是快取共用的，呼叫端必須先 .copy() 再進行任何 paste()/原地修改
    （目前所有呼叫端都是這樣做的）。
    """
    base_image = Image.open(base_image_path).convert("RGBA")
    return base_image.resize(base_size, Image.Resampling.LANCZOS)


@functools.lru_cache(maxsize=None)
def _load_edge_enhanced_base_image(base_image_path: str, base_size: tuple) -> Image.Image:
    """`add_transport_icon_to_image` 專用：多做一次邊緣強化（FIND_EDGES + ImageChops.add），
    結果同樣依 (路徑, 尺寸) 快取。
    """
    base_image = Image.open(base_image_path).convert("RGBA")
    edges = base_image.filter(ImageFilter.FIND_EDGES)
    base_image = ImageChops.add(base_image, edges, scale=1.0, offset=0)
    return base_image.resize(base_size, Image.Resampling.LANCZOS)


def _show_cordinate(combined_image: Image.Image) -> None:
    """顯示格線座標，方便人工確認圖示位置（除錯用；目前沒有任何正式呼叫端會傳
    show_cordinate=True，屬於未被觸發的除錯路徑）。

    2026-07-24 修正紀錄（沿用自重構前的舊邏輯，原本有三個問題）：
    1. 原本 plt.figure(figsize=(100, 100)) 是在 plt.imshow() 之後才建立，等於開了一張
       全新的空白大圖，真正畫了圖的反而是先前的預設 figure；改成先建立 figure 再畫圖。
    2. 原本會把圖片存成暫存檔 "test.png" 再讀回來；改成直接把 PIL Image 傳給 imshow()，
       不需要借助硬碟暫存檔（也不會有覆蓋同名檔案、從未清除的問題）。
    3. 加上 plt.close(fig)，避免長迴圈裡重複呼叫時 matplotlib figure 持續累積造成記憶體洩漏。
    """
    fig = plt.figure(figsize=(100, 100))
    plt.imshow(combined_image)
    plt.xticks(np.arange(0, 1100, 100))
    plt.yticks(np.arange(0, 1100, 100))
    plt.show()
    plt.close(fig)


def add_icon_to_image(base_image_path, data, show_cordinate=False, base_size=(1000, 1000)):
    """在底圖上加入碰撞圖示（`generate_v1`/`generate_v2` 使用，資料無受傷人數欄位）。"""
    base_image = _load_base_image(base_image_path, base_size)
    combined_image = base_image.copy()

    for icon_image_path, position, angle, loc_name, coll_name in data:
        icon_image = white_to_transparent(icon_image_path)
        width, length = icon_image.size
        if length > ICON_RESIZE_THRESHOLD or width > ICON_RESIZE_THRESHOLD:
            length //= ICON_SHRINK_DIVISOR
            width //= ICON_SHRINK_DIVISOR
            icon_image = icon_image.resize((length, width), Image.Resampling.LANCZOS)

        r_icon_image = icon_image.rotate(-angle, expand=True)
        r_icon_image = crop_transparent(r_icon_image)
        combined_image.paste(r_icon_image, position, r_icon_image)

    if show_cordinate:
        _show_cordinate(combined_image)
    return combined_image


def add_transport_icon_to_image(base_image_path, data, show_cordinate=False, base_size=(1000, 1000)):
    """在底圖上加入碰撞圖示（`generate_mix`/`generate_tune_v1`/`generate_tune_v2` 的運研所
    分支使用，底圖會加上邊緣強化效果）。

    data 每個項目格式為 (overlap, icon_image_path, position, angle, loc_name, coll_name)；
    overlap 是「連續事件數」，1 代表單純貼一次，大於 1 時會沿對角線疊貼多次。
    （2026-07-24：這裡原本是 add_transport_icon_to_image() / add_transport_icon_to_image_overlap()
    兩個幾乎一樣的函式，合併成這一個；不需要疊貼的呼叫端固定傳 overlap=1 即可。）
    """
    base_image = _load_edge_enhanced_base_image(base_image_path, base_size)
    combined_image = base_image.copy()

    for overlap, icon_image_path, position, angle, loc_name, coll_name in data:
        icon_image = white_to_transparent(icon_image_path)
        width, length = icon_image.size
        if length > ICON_RESIZE_THRESHOLD or width > ICON_RESIZE_THRESHOLD:
            length //= TRANSPORT_ICON_SHRINK_DIVISOR
            width //= TRANSPORT_ICON_SHRINK_DIVISOR
            icon_image = icon_image.resize((length, width), Image.Resampling.LANCZOS)

        r_icon_image = icon_image.rotate(-angle, expand=True)

        width, length = r_icon_image.size
        length //= TRANSPORT_ICON_SECOND_SHRINK_RATIO
        width //= TRANSPORT_ICON_SECOND_SHRINK_RATIO
        r_icon_image = r_icon_image.resize((int(length), int(width)), Image.Resampling.LANCZOS)

        r_icon_image = crop_transparent(r_icon_image)

        if overlap > 1:
            basis_pos = (position[0] - int(overlap / 2) * 10, position[1] - int(overlap / 2) * 10)
            for i in range(overlap):
                move_pos = (basis_pos[0] + 10 * i, basis_pos[1] + 10 * i)
                combined_image.paste(r_icon_image, move_pos, r_icon_image)
        else:
            combined_image.paste(r_icon_image, position, r_icon_image)

    if show_cordinate:
        _show_cordinate(combined_image)
    return combined_image


def _draw_hurt_number(icon_image: Image.Image, angle: int, hurt_num: int) -> Image.Image:
    """在圖示中心畫上受傷人數數字，並依旋轉角度微調文字位置。"""
    # 如果受傷人數為 0，直接回傳原圖，不標註數字
    if hurt_num == 0:
        return icon_image

    # 準備在圖像上畫數字
    draw = ImageDraw.Draw(icon_image)
    text = str(hurt_num)
    # 計算數字的邊界框
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)  # (舊版)
    # left, top, right, bottom = draw.textbbox((0, 0), text, font=big_font)
    text_width = right - left
    text_height = bottom - top

    # 圖示中心點座標
    image_center_x = icon_image.size[0] / 2
    image_center_y = icon_image.size[1] / 2
    text_position = (int(image_center_x - text_width / 2), int(image_center_y - text_height / 2))

    # 根據旋轉角度微調文字位置
    # if angle == -90:
    #     text_position_bias = (text_position[0] + 1, text_position[1] - 3)
    # elif angle == 90:
    #     text_position_bias = (text_position[0] - 1, text_position[1] - 3)
    # elif angle == 180:
    #     text_position_bias = (text_position[0], text_position[1] - 4)
    # else:
    #     text_position_bias = (text_position[0] + 2, text_position[1] - 2)

    # 自動化調整位置：根據角度將基準點與旋轉向量進行微偏移，減少手動參數
    # 設定調整幅度（像素）：根據數字高度和寬度適當調整
    bias_magnitude = max(2, int(0.03 * max(text_width, text_height)))
    # 角度轉換為徑度
    theta = math.radians(angle)
    # 根據旋轉角度，自動向逆方向微調數字放置位置
    dx = -int(bias_magnitude * math.sin(theta))
    dy = -int(bias_magnitude * math.cos(theta))
    text_position_bias = (text_position[0] + dx, text_position[1] + dy)
        
    # 將受傷人數畫在調整後的位置，黑色顏色
    draw.text(text_position_bias, text, font=font, fill=(0, 0, 0))   # (舊版)
    # draw.text(text_position_bias, text, font=big_font, fill=(0, 0, 0))
    return icon_image


def _paste_debug_preview(combined_image, position, icon_image):
    """除錯用：在真正位置左側貼一份旋轉/縮放前的原圖預覽（紅框）。"""
    pas_icon_image = deepcopy(icon_image)
    pas_icon_image = crop_transparent(pas_icon_image)
    pas_icon_image = ImageOps.expand(pas_icon_image, border=1, fill=(255, 0, 0, 255))
    temp_x, temp_y = position[0] - 100, position[1]
    combined_image.paste(pas_icon_image, (temp_x, temp_y), pas_icon_image)
    return combined_image


def _paste_debug_annotated(combined_image, position, cropped_image, angle, loc_name, way_direction, orign_description):
    """除錯用：在真正位置貼上處理後的圖示（藍框）並標註方向/地點文字說明。"""
    cropped_image = ImageOps.expand(cropped_image, border=1, fill=(0, 0, 255, 255))
    combined_image.paste(cropped_image, position, cropped_image)

    cropped_center_x = cropped_image.size[0] // 2
    cropped_center_y = cropped_image.size[1] // 2
    draw = ImageDraw.Draw(combined_image)
    x, y = (position[0] + cropped_center_x, position[1] + cropped_center_y)
    orign_descr = orign_description.replace("，", "，\n")
    descr = (str(angle) + way_direction).replace("，", "，\n")

    if "\n" in descr:
        draw.text((x - cropped_center_x - 20, y - cropped_center_y - 80), orign_descr, font=angle_font, fill=(255, 0, 0))
        draw.text((x - cropped_center_x - 20, y - cropped_center_y - 40), descr, font=angle_font, fill=(0, 0, 255))
    else:
        draw.text((x - cropped_center_x - 20, y - cropped_center_y - 40), orign_descr, font=angle_font, fill=(255, 0, 0))
        draw.text((x - cropped_center_x - 20, y - cropped_center_y - 20), descr, font=angle_font, fill=(0, 0, 255))
    draw.text((x - cropped_center_x, y + cropped_center_y), loc_name, font=angle_font, fill=(0, 0, 255))
    return combined_image


def add_number_icon_to_image(base_image_path, data, debug=False, show_cordinate=False, base_size=(1000, 1000)):
    """在底圖上加入碰撞圖示並標註受傷人數（`generate_v3`/`generate_mix`/`generate_tune_v1`/
    `generate_tune_v2` 的阡陌分支使用）。

    data 每個項目格式為 (overlap, icon_image_path, orign_pos, position, angle, hurt_num,
    loc_name, main_collision, collision_label, way_direction, orign_description)；overlap
    語意同 add_transport_icon_to_image()。
    （2026-07-24：這裡原本是 add_number_icon_to_image() / add_number_icon_to_image_overlap()
    兩個幾乎一樣的函式，合併成這一個；不需要疊貼的呼叫端固定傳 overlap=1 即可。）
    """
    base_image = _load_base_image(base_image_path, base_size)
    combined_image = base_image.copy()

    for (overlap, icon_image_path, orign_pos, position, angle, hurt_num, loc_name, main_collision,
         collision_label, way_direction, orign_description) in data:
        icon_image = white_to_transparent(icon_image_path)
        width, length = icon_image.size
        if length > ICON_RESIZE_THRESHOLD or width > ICON_RESIZE_THRESHOLD:
            ###################################################
            # 舊版
            length //= ICON_SHRINK_DIVISOR
            width //= ICON_SHRINK_DIVISOR
            # 若圖示尺寸超過 ICON_RESIZE_THRESHOLD，將其寬高縮小 ICON_SHRINK_DIVISOR 倍

            ###################################################
            # 新版放大
            # length //= int(ICON_SHRINK_DIVISOR / 1.5)
            # width //= int(ICON_SHRINK_DIVISOR / 1.5)
            ###################################################
     
            icon_image = icon_image.resize((length, width), Image.Resampling.LANCZOS)

        if debug:
            combined_image = _paste_debug_preview(combined_image, position, icon_image)

        icon_image = icon_image.rotate(-angle, expand=True)
        icon_image = _draw_hurt_number(icon_image, angle, hurt_num)
        cropped_image = icon_image

        if debug:
            combined_image = _paste_debug_annotated(
                combined_image, position, cropped_image, angle, loc_name, way_direction, orign_description
            )
        else:
            width, length = cropped_image.size
            ###################################################
            # 舊版
            length //= NUMBER_ICON_SHRINK_RATIO
            width //= NUMBER_ICON_SHRINK_RATIO        
            ###################################################
            # 新版
            # length //= 1
            # width //= 1        
            ###################################################
            cropped_image = cropped_image.resize((int(length), int(width)), Image.Resampling.LANCZOS)

            if overlap > 1:
                basis_pos = (position[0] - int(overlap / 2) * 10, position[1] - int(overlap / 2) * 10)
                for i in range(overlap):
                    move_pos = (basis_pos[0] + 10 * i, basis_pos[1] + 10 * i)
                    combined_image.paste(cropped_image, move_pos, cropped_image)
            else:
                combined_image.paste(cropped_image, position, cropped_image)

    if show_cordinate:
        _show_cordinate(combined_image)
    return combined_image
