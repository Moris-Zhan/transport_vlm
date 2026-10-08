"""通用影像處理小工具：透明背景裁切、白底轉透明。"""

import functools

from PIL import Image


def crop_transparent(image: Image.Image) -> Image.Image:
    """根據 alpha channel 自動裁切圖片的透明邊界。

    參數:
        image (PIL.Image.Image): 輸入的影像 (需有透明通道，如 RGBA)

    回傳:
        PIL.Image.Image: 已裁切透明邊界後的影像；若無非透明像素則回傳原圖
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = image.split()[-1]
    bbox = alpha.getbbox()

    if bbox:
        return image.crop(bbox)
    return image


@functools.lru_cache(maxsize=None)
def _load_rgba(image_path: str) -> Image.Image:
    """讀檔+轉 RGBA 的快取層。

    合成資料時，同一批碰撞圖示檔案（images/collision_icons(運研所|阡陌)/ 底下固定的幾百個
    PNG）會在整批資料生成過程中被重複開啟成千上萬次；maxsize=None 沒問題，因為 key 的數量
    上限就是硬碟上實際存在的圖片檔案數（幾百張），不會無限成長。

    回傳的 Image 物件是快取共用的，呼叫端不可對它做原地修改（例如直接 ImageDraw.Draw 畫
    東西）——目前所有呼叫端都是先 .resize()/.rotate() 產生新物件後才進一步處理，是安全的。
    """
    return Image.open(image_path).convert("RGBA")


def white_to_transparent(image_path, output_path: str = "") -> Image.Image:
    """開啟圖片並轉成 RGBA（含透明度通道），可選擇另存輸出檔。"""
    img = _load_rgba(image_path)
    if output_path:
        img.save(output_path, "PNG")
    return img
