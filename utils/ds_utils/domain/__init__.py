"""純規則表與純函式：碰撞型態規則、地點/座標對應、幾何計算。

這個套件底下的模組本身的程式碼不做任何檔案或網路 I/O。但這不代表可以「單獨匯入」而不依賴
images/ 底下的實際素材：匯入 domain/* 底下任何模組時，Python 會先匯入上層套件
utils.ds_utils（也就是跑一次 utils/ds_utils/__init__.py），而該檔案會匯入
utils/ds_utils/assets.py；assets.py 在模組層級就會建立圖示目錄快取（例如
TRANSPORT_THIN_COLLISIONS = _build_transport_thin_collisions()），這會實際掃描
images/collision_icons(...)/ 底下的真實資料夾。因此即使 domain/* 本身的邏輯是純函式，
單獨匯入例如 utils.ds_utils.domain.geometry 仍然需要工作目錄下存在真正的 images/ 素材樹，
否則會在匯入階段就拋出 FileNotFoundError。真正需要讀檔（底圖 CSV、圖示檔查找）的邏輯本體
在 utils/ds_utils/assets.py；若要讓 domain/* 真正可以脫離 images/ 素材單獨匯入/測試，需要
把 assets.py 的目錄掃描改成延遲（lazy）建立，這是另一個較大的後續項目，不在本次修正範圍。
"""
