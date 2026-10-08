"""合成交通事故碰撞構圖的資料產生模組。

`CollisionDiagram` 保留與重構前完全相同的公開介面（`from utils.ds_utils import
CollisionDiagram`），內部依功能拆成：

- `base.py`      共用的底圖/碰撞型態/座標選擇邏輯（`CollisionDiagramBase`）
- `rendering.py`  把圖示合成到底圖上的繪製函式
- `image_ops.py`  影像裁切/透明背景等小工具
- `io_utils.py`   JSONL 輸出
- `generators/`   6 個資料格式版本，各自一個子類別：
    - `CollisionDescriptionV1/V2/V3`：肇事型態描述 Ver.1（路段+肇事類型）/ Ver.2（+行車方向）/
      Ver.3（+傷亡人數）
    - `CollisionDiagramMix`、`CollisionDiagramTuneV1`、`CollisionDiagramTuneV2`
- `facade.py`     把 6 個版本合併回 `CollisionDiagram` 供既有程式碼相容使用
"""

from utils.ds_utils.assets import IconAssetNotFoundError
from utils.ds_utils.base import CollisionDiagramBase
from utils.ds_utils.facade import CollisionDiagram
from utils.ds_utils.generators import (
    CollisionDescriptionV1,
    CollisionDescriptionV2,
    CollisionDescriptionV3,
    CollisionDiagramMix,
    CollisionDiagramTuneV1,
    CollisionDiagramTuneV2,
)

__all__ = [
    "CollisionDiagram",
    "CollisionDiagramBase",
    "CollisionDescriptionV1",
    "CollisionDescriptionV2",
    "CollisionDescriptionV3",
    "CollisionDiagramMix",
    "CollisionDiagramTuneV1",
    "CollisionDiagramTuneV2",
    "IconAssetNotFoundError",
]
