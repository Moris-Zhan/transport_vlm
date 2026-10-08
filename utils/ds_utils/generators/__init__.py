"""各資料格式版本的產生器類別，一個版本一支子類別，共用 `CollisionDiagramBase`。"""

from utils.ds_utils.generators.description_v1 import CollisionDescriptionV1
from utils.ds_utils.generators.description_v2 import CollisionDescriptionV2
from utils.ds_utils.generators.description_v3 import CollisionDescriptionV3
from utils.ds_utils.generators.mix import CollisionDiagramMix
from utils.ds_utils.generators.tune_v1 import CollisionDiagramTuneV1
from utils.ds_utils.generators.tune_v2 import CollisionDiagramTuneV2

__all__ = [
    "CollisionDescriptionV1",
    "CollisionDescriptionV2",
    "CollisionDescriptionV3",
    "CollisionDiagramMix",
    "CollisionDiagramTuneV1",
    "CollisionDiagramTuneV2",
]
