from fastapi import FastAPI, Query, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from typing import Optional
import base64
import argparse, uvicorn
from io import StringIO
from openai import OpenAI
import uuid
from datetime import datetime
from html_templates import HOME_PAGE_TEMPLATE, RESULT_PAGE_TEMPLATE
import time
import webbrowser
import asyncio
from threading import Lock

app = FastAPI(title="Chat Completions API")

# --------------------------------------------------------------------------- #
# OpenAI client Configuration
# --------------------------------------------------------------------------- #
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
client = OpenAI(base_url=API_BASE_URL,api_key=API_KEY)
model = client.models.list().data[0].id
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

def encode_image(image_bytes: bytes) -> str:
    """將上傳圖片轉換成 base64 格式字串"""
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")


@app.post("/vlm_response")
async def vlm_response(
    prompt: str =  Form("你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖，對路口進行分析並辨識肇事型態。", 
                            description="詢問碰撞構圖VLM助理問題。"),
    file: Optional[UploadFile] = File(..., description="上傳的圖片檔案") ,
    temperature: float = Form(0.15, description="控制模型生成的隨機性。較低的溫度（例如 0.2）會讓模型會選擇機率較高的詞；較高的溫度（例如 1.0 或以上）會讓生成的詞彙更具隨機性和多樣性。"),
    top_p: float = Form(0.9, description="控制生成的隨機性，通常設置值在0到1之間，數值越小，生成的內容越保守且可預測；數值越大，生成的內容越具有創造性與隨機性"),
    max_tokens: int = Form(1000, description="設置生成過程中最多生成Token的數量。")
):
    """
    碰撞構圖VLM API接口
    
    支援:
    - 文字輸入
    - 圖片上傳(可選)
    - 自訂temperature和max_tokens
    """
    
    try:
        image_bytes = await file.read()
        print(f"接收圖片大小: {len(image_bytes)} bytes")
        print(f"圖片類型: {file.content_type}")
        
        # 轉換為 base64
        image_data = encode_image(image_bytes)    

        start_time = time.time()   

        stream = client.chat.completions.create(
                model=client.models.list().data[0].id,
                messages= [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":  prompt},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                },
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,             
                stream=True
            )
        f = StringIO()
        for chunk in stream:
            f.write(chunk.choices[0].delta.content)
            print(chunk.choices[0].delta.content, end="")
        print()

        # 計算推理時間
        inference_time = time.time() - start_time
        print(f"\n推理時間: {inference_time:.2f} 秒")
        
        # 暫時返回模擬響應(請替換為實際的API調用)
        result = {
            "status": "success",
            "response": f.getvalue(),
            "inference_time": f"{inference_time:.2f}s",
            "received_params": {
                "model": model,  
                "temperature": temperature,  
                "max_tokens": max_tokens     
            },
            "image_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(image_bytes),
            }
        }        
        return JSONResponse(content=result)
        
    except Exception as e:
        result = {
            "status": "failed",       
        }        
        return JSONResponse(content=result)
        raise HTTPException(status_code=500, detail=f"處理請求時發生錯誤: {str(e)}")


def call_model(prompt, image_data, temperature, top_p, max_tokens):
    """
    封裝同步的 client.chat 呼叫
    讓 asyncio 可以透過 run_in_executor 平行執行
    """
    response = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages= [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":  prompt},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            },
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,             
            stream=False
        )

    result = response.choices[0].message.content
    return result

@app.post("/multi_response")
async def multi_response(
    prompt: str =  Form("你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖，對路口進行分析並辨識肇事型態。", 
                            description="詢問碰撞構圖VLM助理問題。"),
    file: Optional[UploadFile] = File(..., description="上傳的圖片檔案") ,
    temperature: float = Form(0.15, description="控制模型生成的隨機性。較低的溫度（例如 0.2）會讓模型會選擇機率較高的詞；較高的溫度（例如 1.0 或以上）會讓生成的詞彙更具隨機性和多樣性。"),
    top_p: float = Form(0.9, description="控制生成的隨機性，通常設置值在0到1之間，數值越小，生成的內容越保守且可預測；數值越大，生成的內容越具有創造性與隨機性"),
    max_tokens: int = Form(1000, description="設置生成過程中最多生成Token的數量。"),
    threads: int = Form(10, description="設置平行呼叫client.chat產生結果的數量")
):
    """
    碰撞構圖VLM API接口
    
    支援:
    - 文字輸入
    - 圖片上傳(可選)
    - 自訂temperature和max_tokens
    """
    
    try:
        image_bytes = await file.read()
        print(f"接收圖片大小: {len(image_bytes)} bytes")
        print(f"圖片類型: {file.content_type}")
        
        # 轉換為 base64
        image_data = encode_image(image_bytes)    
        start_time = time.time()         
        loop = asyncio.get_event_loop()

        # === 建立 threads 個平行任務 ===
        tasks = [
            loop.run_in_executor(
                None,
                call_model,
                prompt,
                image_data,
                temperature,
                top_p,
                max_tokens
            )
            for _ in range(threads)
        ]

        # 等待所有平行任務完成
        all_results = await asyncio.gather(*tasks)

        # 計算推理時間
        inference_time = time.time() - start_time
        print(f"\n推理時間: {inference_time:.2f} 秒")
        
        # 暫時返回模擬響應(請替換為實際的API調用)
        result = {
            "status": "success",
            "response": all_results,
            "inference_time": f"{inference_time:.2f}s",
            "received_params": {
                "model": model,  
                "temperature": temperature,  
                "max_tokens": max_tokens     
            },
            "image_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(image_bytes),
            }
        }        
        return JSONResponse(content=result)
        
    except Exception as e:
        result = {
            "status": "failed",       
        }        
        return JSONResponse(content=result)

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
    """處理分析請求並生成結果頁面"""
    print(f"max_tokens: {max_tokens}")
    print(f"top_p: {top_p}")
    print(f"temperature: {temperature}")
    print(f"file: {file.filename}")
    
    try:
        # 讀取圖片
        image_bytes = await file.read()
        content_type = file.content_type or "image/png"
        
        if content_type.startswith("image/"):
            image_data = f"data:{content_type};base64," + base64.b64encode(image_bytes).decode("utf-8")
        else:
            image_data = "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

        start_time = time.time()
        # 調用 VLM API
        stream = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                },
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True
        )
        
        f = StringIO()
        for chunk in stream:
            if chunk.choices[0].delta.content:
                f.write(chunk.choices[0].delta.content)
                print(chunk.choices[0].delta.content, end="")
        print()
        # 計算推理時間
        inference_time = time.time() - start_time
        print(f"\n推理時間: {inference_time:.2f} 秒")
        
        response_text = f.getvalue()
        
        # 生成結果 ID 並暫存
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results_cache.set(result_id, {
            "image_data": image_data,
            "filename": file.filename,
            "size": len(image_bytes),
            "response": response_text,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timestamp": timestamp ,
            "timestamp_safe": timestamp.replace(":", "-"),  # 用於檔案名稱
            "inference_time": f"{inference_time:.2f}"  # 新增推理時間
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
    # API 端點: http://localhost:8080/vlm_response
    # Swagger 文件: http://localhost:8080/docs

    # https://afspod-services.ai.foxconn.com/a4665b22-6514-463b-8dd3-28ccb78ae0b7/moris/
