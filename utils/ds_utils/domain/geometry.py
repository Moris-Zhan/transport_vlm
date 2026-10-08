"""路口地點/座標對應表，以及碰撞圖示重疊偵測用的幾何計算（純函式，不做任何 I/O）。"""

import numpy as np

# 順時針方向
way_angle = {
    0: "北",
    90: "東",
    180: "南",
    270: "西"
}

inverse_angle_mapping = {v: k for k, v in way_angle.items()}

location_csv = "底圖info_v7.csv"
# 21種地點對應表（原始名稱 → 中文友善名稱） => 5*4+1
location_mapping = {
    "路口中端": "路口中端",

    "路段-西-往西": "西側路段(往西)",
    "路段-西-往東": "西側路段(往東)",
    "鄰近路口-西": "西側鄰近路口",
    "路口近端-西": "西側路口近端",
    "路口遠端-西": "西側路口遠端",

    "路段-東-往東": "東側路段(往東)",
    "路段-東-往西": "東側路段(往西)",
    "鄰近路口-東": "東側鄰近路口",
    "路口近端-東": "東側路口近端",
    "路口遠端-東": "東側路口遠端",

    "路段-南-往南": "南側路段(往南)",
    "路段-南-往北": "南側路段(往北)",
    "鄰近路口-南": "南側鄰近路口",
    "路口近端-南": "南側路口近端",
    "路口遠端-南": "南側路口遠端",

    "路段-北-往北": "北側路段(往北)",
    "路段-北-往南": "北側路段(往南)",
    "鄰近路口-北": "北側鄰近路口",
    "路口近端-北": "北側路口近端",
    "路口遠端-北": "北側路口遠端",
}

rotation_angle_mapping = {
    "路口中端": 0,

    "路段-西-往西": -90,
    "路段-西-往東": 90,
    "鄰近路口-西": 90,
    "路口近端-西": 90,
    "路口遠端-西": -90,

    "路段-東-往東": 90,
    "路段-東-往西": -90,
    "鄰近路口-東": -90,
    "路口近端-東": 90,
    "路口遠端-東": 90,

    "路段-南-往南": 180,
    "路段-南-往北": 0,
    "鄰近路口-南": 0,
    "路口近端-南": 0,
    "路口遠端-南": 180,

    "路段-北-往北": 0,
    "路段-北-往南": 180,
    "鄰近路口-北": 180,
    "路口近端-北": 180,
    "路口遠端-北": 0,
}

# 所有可能的地點欄位
locations = list(rotation_angle_mapping.keys())

inverse_location_mapping = {v: k for k, v in location_mapping.items()}


def compute_iou(boxA, boxB):
    # box 格式: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# ========== 方案 2: 使用 NumPy 向量化 (適合大量數據,最快) ==========
def compute_iou_vectorized(curr_box, prev_boxes):
    """向量化計算 curr_box 與所有 prev_boxes 的 IOU"""
    # 假設 box 格式為 [x1, y1, x2, y2]
    curr_box = np.array(curr_box)
    prev_boxes = np.array(prev_boxes)

    # 計算交集區域
    x1 = np.maximum(curr_box[0], prev_boxes[:, 0])
    y1 = np.maximum(curr_box[1], prev_boxes[:, 1])
    x2 = np.minimum(curr_box[2], prev_boxes[:, 2])
    y2 = np.minimum(curr_box[3], prev_boxes[:, 3])

    # 交集面積
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 各自面積
    curr_area = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
    prev_areas = (prev_boxes[:, 2] - prev_boxes[:, 0]) * (prev_boxes[:, 3] - prev_boxes[:, 1])

    # IOU
    union = curr_area + prev_areas - intersection
    iou = intersection / (union + 1e-6)

    return iou
