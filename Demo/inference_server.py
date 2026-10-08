from fastapi import FastAPI, Query, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Optional
import base64
import argparse, uvicorn
import random
from pathlib import Path
import uuid
from datetime import datetime
from html_templates import HOME_PAGE_TEMPLATE, RESULT_PAGE_TEMPLATE
import time
import webbrowser
from threading import Lock

from image_lookup import build_index, find_best_match, DEFAULT_THRESHOLD

app = FastAPI(title="Chat Completions API")

# --------------------------------------------------------------------------- #
# 歷史推論結果比對索引 (pHash) —— 取代即時 VLM 呼叫
# --------------------------------------------------------------------------- #
JSONL_PATH = Path(__file__).parent / "test.jsonl"
SIMILARITY_THRESHOLD = DEFAULT_THRESHOLD
match_index = build_index(str(JSONL_PATH))
SIMULATED_PROCESSING_DELAY_RANGE = (2.0, 3.0)  # 模擬 VLM 推理耗時，讓比對結果不會秒回
# --------------------------------------------------------------------------- #
# 用於暫存結果的字典 (生產環境建議使用 Redis 或資料庫)
class ResultCache:
    """線程安全的結果快取管理器"""
    
    def __init__(self):
        self._cache = {}
        self._lock = Lock()
    
    def set(self, key: str, value: dict) -> None:
        """設置快取值"""
        with self._lock:
            self._cache[key] = value
    
    def get(self, key: str) -> Optional[dict]:
        """獲取快取值"""
        with self._lock:
            return self._cache.get(key)
    
    def exists(self, key: str) -> bool:
        """檢查鍵是否存在"""
        with self._lock:
            return key in self._cache
    
    def delete(self, key: str) -> None:
        """刪除快取值"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """清空所有快取"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """返回快取大小"""
        with self._lock:
            return len(self._cache)

# 初始化結果快取
results_cache = ResultCache()


@app.get("/", response_class=HTMLResponse)
async def main():
    results_cache.clear()
    return HTMLResponse(content=HOME_PAGE_TEMPLATE)


@app.post("/analyze")
async def analyze(
    prompt: str =  Form("你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖，對路口進行分析並辨識肇事型態。", 
                            description="詢問碰撞構圖VLM助理問題。"),
    file: Optional[UploadFile] = File(..., description="上傳的圖片檔案") ,
    temperature: float = Form(0.15, description="控制模型生成的隨機性。較低的溫度（例如 0.2）會讓模型會選擇機率較高的詞；較高的溫度（例如 1.0 或以上）會讓生成的詞彙更具隨機性和多樣性。"),
    top_p: float = Form(0.9, description="控制生成的隨機性，通常設置值在0到1之間，數值越小，生成的內容越保守且可預測；數值越大，生成的內容越具有創造性與隨機性"),
    max_tokens: int = Form(1000, description="設置生成過程中最多生成Token的數量。")
):
    """處理分析請求並生成結果頁面（以 pHash 比對 test.jsonl 歷史資料，不呼叫即時 VLM）"""

    try:
        # 讀取圖片
        image_bytes = await file.read()
        content_type = file.content_type or "image/png"

        if content_type.startswith("image/"):
            image_data = f"data:{content_type};base64," + base64.b64encode(image_bytes).decode("utf-8")
        else:
            image_data = "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

        start_time = time.time()
        time.sleep(random.uniform(*SIMULATED_PROCESSING_DELAY_RANGE))  # 模擬模型處理時間
        match = find_best_match(image_bytes, match_index, SIMILARITY_THRESHOLD)
        lookup_time = time.time() - start_time

        if match is not None:
            response_text = (
                # f"✅ 比對來源: {match.record['image']}（雜湊距離: {match.distance}）\n\n"
                f"{match.record['inference']}"
            )
        else:
            response_text = (
                f"⚠️ 查無此紀錄：找不到相似度足夠的歷史圖片"
                f"（比對門檻: {SIMILARITY_THRESHOLD}, 資料庫: {len(match_index)} 筆）"
            )

        # 生成結果 ID 並暫存
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results_cache.set(result_id, {
            "image_data": image_data,
            "filename": file.filename,
            "size": len(image_bytes),
            "response": response_text,
            # "model": "本地圖片比對 (pHash)",
            "model": "Mistral-Small-3.2-24B-Instruct-2506",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timestamp": timestamp ,
            "timestamp_safe": timestamp.replace(":", "-"),  # 用於檔案名稱
            "inference_time": f"{lookup_time:.3f}"  # 比對耗時
        })

        result_url = f"/result/{result_id}"
        print(result_url)

        # time.sleep(0.5)
        # webbrowser.open(result_url)
        # 重定向到結果頁面
        return RedirectResponse(url=result_url, status_code=303)
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"處理請求時發生錯誤: {str(e)}")


@app.get("/result/{result_id}", response_class=HTMLResponse)
async def show_result(result_id: str):
    """顯示分析結果頁面"""
    if not results_cache.exists(result_id):
        raise HTTPException(status_code=404, detail="結果不存在或已過期")
    
    data = results_cache.get(result_id)    
    
    # 使用模板並填入資料
    html_content = RESULT_PAGE_TEMPLATE.format(
        timestamp=data['timestamp'],
        image_data=data['image_data'],
        filename=data['filename'],
        size=data['size'],
        response=data['response'],
        model=data['model'],
        temperature=data['temperature'],
        top_p=data['top_p'],
        max_tokens=data['max_tokens'],
        timestamp_safe=data['timestamp_safe'],
        inference_time=data['inference_time']  # 新增推理時間
    )

    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    # parser.add_argument("--host", type=str, default="https://afspod-services.ai.foxconn.com/a4665b22-6514-463b-8dd3-28ccb78ae0b7/moris/")
    parser.add_argument("--port", type=int, default=8080)
    # parser.add_argument("--port", type=int, default=10083)

    args = parser.parse_args()
    uvicorn.run("inference_server:app", host=args.host, port=args.port, log_level="info", workers=1)

    # 使用方式:
    # python inference_server.py --port 8080
    #
    # 網頁介面: http://localhost:8080/
    # Swagger 文件: http://localhost:8080/docs

    # https://afspod-services.ai.foxconn.com/a4665b22-6514-463b-8dd3-28ccb78ae0b7/moris/
