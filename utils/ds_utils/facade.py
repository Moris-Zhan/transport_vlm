"""向後相容用的合併類別。

歷史上曾有多支呼叫端（`make_data.py`、`make_data_v1.py`，以及已於 2026-07-23 移除的
`utils/make_utils/make_data_v1~v4.py`）對同一個 `CollisionDiagram` 實例呼叫不同的
`generate*` 方法。拆成一版一個子類別後，這個類別把全部 6 個版本用 mixin 的方式合併回
同一個名稱，供 `make_data.py`/`make_data_v1.py` 或任何想直接用 `CollisionDiagram` 的
呼叫端使用；若只需要單一版本，改用 `make_data.py --generator <版本名稱>` 或直接從
`utils.ds_utils.generators` import 對應子類別（例如 `CollisionDiagramTuneV2`）即可。
"""

from utils.ds_utils.generators import (
    CollisionDescriptionV1,
    CollisionDescriptionV2,
    CollisionDescriptionV3,
    CollisionDiagramMix,
    CollisionDiagramTuneV1,
    CollisionDiagramTuneV2,
)


class CollisionDiagram(
    CollisionDescriptionV1,
    CollisionDescriptionV2,
    CollisionDescriptionV3,
    CollisionDiagramMix,
    CollisionDiagramTuneV1,
    CollisionDiagramTuneV2,
):
    """合併全部 6 個 generate* 版本方法的相容類別，行為與重構前的 `CollisionDiagram` 相同。"""

    pass
