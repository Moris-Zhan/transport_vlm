import requests

# FastAPI 服務端點
url = "https://afspod-services.ai.foxconn.com/a4665b22-6514-463b-8dd3-28ccb78ae0b7/moris/vlm_response"

# 要上傳的圖片路徑
image_path = "data/collision_diagram/synthetic_tune/test/Image_1761805674042535.png"  # 改成你實際的測試圖片檔案

# 上傳圖片檔案
files = {
    "file": (image_path, open(image_path, "rb"), "image/png")
}

try:
    print(f"📤 正在發送請求到 {url} ...")
    response = requests.post(url, files=files, timeout=120)

    # 顯示結果
    print(f"📥 狀態碼: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("\n✅ 響應結果：")
        print(result)
    else:
        print("\n❌ 發生錯誤：")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"⚠️ 請求失敗: {e}")
