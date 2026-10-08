from PIL import Image

import base64
import mimetypes
from pathlib import Path
from pydantic import BaseModel
from typing import List
import json
import shutil
import base64
import mimetypes
from pathlib import Path
import enum
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_show(image):
    img = mpimg.imread(image)
    height, width = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)  # 單一子圖

    ax.imshow(img)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    # 文字顯示在 x 軸標籤位置
    # text_str = str(text)  # 將樣本物件轉成文字
    # ax.set_xlabel(text_str, fontsize=12, labelpad=20)

    plt.show()


def prepare_directory(dir_path):
    path = Path(dir_path)
    if path.exists():
        # 清除目錄內容但保留目錄本身
        [f.unlink() for f in path.glob('*') if f.is_file()]
        [shutil.rmtree(d) for d in path.glob('*') if d.is_dir()]
    else:
        path.mkdir(parents=True, exist_ok=True)

font_path = 'Microsoft-JhengHei.ttf'  # 字型檔完整路徑
prop = fm.FontProperties(fname=font_path)        

"""
OpenAI Vision API 官方說明要求傳入的圖片格式必須是 JPEG、PNG、WEBP，
且圖片本身應該是在 RGB 或 RGBA 色彩空間中，
但沒有特別要求用戶在 encode_image 時針對色彩做轉換，僅需保持影像原格式即可
"""

def encode_image(path: Path) -> str:
    """Return the image as a data‑URL string accepted by the OpenAI Vision API."""
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def custom_process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs