"""
HTML 模板檔案
包含所有 HTML 頁面的模板字串
"""

HOME_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交通事故碰撞構圖分析系統</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Microsoft JhengHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background-color: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: bold;
            font-size: 14px;
        }
        input[type="text"],
        input[type="number"],
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus,
        input[type="number"]:focus,
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        input[readonly],
        textarea[readonly] {
            background-color: #f0f0f0 !important;
            cursor: not-allowed !important;
            color: #666;
        }
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        .file-upload {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px dashed #667eea;
            border-radius: 8px;
            cursor: pointer;
            background-color: #f8f9ff;
        }
        input[type="file"]::-webkit-file-upload-button {
            background-color: #667eea;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin-right: 10px;
        }
        .param-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        .param-item {
            display: flex;
            flex-direction: column;
        }
        .param-item small {
            color: #888;
            margin-top: 5px;
            font-size: 11px;
        }
        button[type="submit"] {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button[type="submit"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        button[type="submit"]:active {
            transform: translateY(0);
        }
        .preview-area {
            margin-top: 15px;
            text-align: center;
        }
        .preview-area img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚦 交通事故碰撞構圖分析</h1>
        <p class="subtitle">上傳事故圖片進行 AI 智能分析</p>
        
        <form id="uploadForm" action="analyze" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="prompt">📝 分析提示詞</label>
                <textarea 
                    name="prompt" 
                    id="prompt" 
                    readonly
                    placeholder="輸入您的問題或使用預設提示詞..."
                >你是一位專精台灣交通工程與交通安全的專家,請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖,對路口進行分析並辨識肇事型態。</textarea>
            </div>
            
            <div class="form-group">
                <label for="file">📸 上傳圖片 (必填)</label>
                <input 
                    type="file" 
                    name="file" 
                    id="file" 
                    accept="image/*" 
                    required 
                    onchange="previewImage(event)"
                >
                <div id="preview" class="preview-area"></div>
            </div>
            
            <div class="form-group">
                <label>⚙️ 模型參數</label>
                <div class="param-grid">
                    <div class="param-item">
                        <input 
                            type="number" 
                            name="temperature" 
                            step="0.01" 
                            min="0" 
                            max="2" 
                            value="0.15"
                            readonly
                        >
                        <small>Temperature</small>
                    </div>
                    <div class="param-item">
                        <input 
                            type="number" 
                            name="top_p" 
                            step="0.01" 
                            min="0" 
                            max="1" 
                            value="0.9"
                            readonly
                        >
                        <small>Top P</small>
                    </div>
                    <div class="param-item">
                        <input 
                            type="number" 
                            name="max_tokens" 
                            min="1" 
                            max="4096" 
                            value="1000"
                            readonly
                        >
                        <small>Max Tokens</small>
                    </div>
                </div>
            </div>
            
            <button type="submit">🚀 開始分析</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>正在分析中,請稍候...</p>
            </div>
        </form>
    </div>
    
    <script>
        function previewImage(event) {
            const preview = document.getElementById('preview');
            const file = event.target.files[0];
            
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.innerHTML = '<img src="' + e.target.result + '" alt="預覽圖片">';
                }
                reader.readAsDataURL(file);
            } else {
                preview.innerHTML = '';
            }
        }
        
        document.getElementById('uploadForm').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
    </script>
</body>
</html>
"""


RESULT_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分析結果 - 交通事故碰撞構圖</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Microsoft JhengHei', Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        h1 {{
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .timestamp {{
            color: #888;
            font-size: 14px;
        }}
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        @media (max-width: 968px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}
        .params-footer {{
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
        }}
        .section {{
            background-color: #f9f9f9;
            border-radius: 12px;
            padding: 25px;
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .image-container {{
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .image-info {{
            margin-top: 15px;
            font-size: 14px;
            color: #666;
        }}
        .params {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        @media (max-width: 768px) {{
            .params {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        .param-item {{
            background-color: white;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .param-label {{
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }}
        .param-value {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .response-section {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffe5a3 100%);
            border-left: 5px solid #ffc107;
            height: fit-content;
        }}
        .response-text {{
            white-space: pre-wrap;
            line-height: 1.8;
            color: #333;
            font-size: 16px;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }}
        .inference-time {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px;
            background-color: rgba(255,255,255,0.5);
            border-radius: 8px;
            font-size: 14px;
        }}
        .time-icon {{
            font-size: 18px;
        }}
        .time-label {{
            color: #666;
            font-weight: 500;
        }}
        .time-value {{
            color: #667eea;
            font-weight: bold;
            font-size: 16px;
        }}
        .actions {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }}
        .btn {{
            padding: 12px 30px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
            border: none;
            font-size: 14px;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .btn-secondary {{
            background-color: #e0e0e0;
            color: #333;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚦 分析結果</h1>
            <p class="timestamp">分析時間: {timestamp}</p>
        </div>
        
        <div class="main-content">
            <div class="section">
                <h2>📸 上傳的圖片</h2>
                <div class="image-container">
                    <img src="{image_data}" alt="交通事故圖片">
                    <div class="image-info">
                        <p><strong>檔案:</strong> {filename}</p>
                        <p><strong>大小:</strong> {size:,} bytes</p>
                    </div>
                </div>
            </div>
            
            <div class="section response-section">
                <h2>🤖 AI 分析結果</h2>
                <div class="response-text">{response}</div>
                <div class="inference-time">
                    <span class="time-icon">⏱️</span>
                    <span class="time-label">推理時間:</span>
                    <span class="time-value">{inference_time} 秒</span>
                </div>
            </div>
        </div>
        
        <div class="params-footer">
            <div class="section">
                <h2>⚙️ 模型參數</h2>
                <div class="params">
                    <div class="param-item">
                        <div class="param-label">模型</div>
                        <div class="param-value">{model}</div>
                    </div>
                    <div class="param-item">
                        <div class="param-label">Temperature</div>
                        <div class="param-value">{temperature}</div>
                    </div>
                    <div class="param-item">
                        <div class="param-label">Top P</div>
                        <div class="param-value">{top_p}</div>
                    </div>
                    <div class="param-item">
                        <div class="param-label">Max Tokens</div>
                        <div class="param-value">{max_tokens}</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="actions">             
            <a href="/" class="btn btn-secondary">🔄 分析新圖片</a>
            <button onclick="window.print()" class="btn btn-secondary">🖨️ 列印結果</button>
            <button onclick="downloadResult()" class="btn btn-secondary">💾 下載報告</button>
        </div>
    </div>
    
    <script>
        function downloadResult() {{
            const content = document.querySelector('.container').innerHTML;
            const blob = new Blob([`
                <!DOCTYPE html>
                <html><head><meta charset="UTF-8"><title>分析結果</title></head>
                <body>${{content}}</body></html>
            `], {{ type: 'text/html' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '交通事故分析_{timestamp_safe}.html';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>
"""
