"""VLM 微調（LoRA/SFT）的可測試邏輯：資源偵測、run 編號管理、資料格式轉換、訓練設定。

`fine_tuning_vlm_trl_traffic_tune_v2.py`（repo 根目錄）是實際執行訓練的 CLI 入口，會匯入
這個套件底下的函式/類別；這裡本身不含任何需要真實 GPU/模型才能執行的邏輯，可以獨立測試。
"""
