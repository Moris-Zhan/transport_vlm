from pydantic import BaseModel
from typing import List
import json
import enum
# vllm 0.28.0 把 GuidedDecodingParams 改名/取代為 StructuredOutputsParams，
# SamplingParams 對應的欄位也從 guided_decoding 改成 structured_outputs
# （舊名在這個版本已經完全移除，不是 deprecated alias，import 會直接
# ImportError；見 vllm/sampling_params.py 的 StructuredOutputsParams）。
from vllm.sampling_params import StructuredOutputsParams
from vllm import LLM, SamplingParams


class Direction(str, enum.Enum):
    STRAIGHT = "直行"
    RIGHT_TURN = "右轉"
    LEFT_TURN = "左轉"
    LEFT_U_TURN = "左迴轉"
    RIGHT_U_TURN = "右迴轉"
    LEFT_CURVE = "左彎"
    RIGHT_CURVE = "右彎"

class Position(str, enum.Enum):
    SOUTH = "南"
    NORTH = "北"
    EAST = "東"
    WEST = "西"

class CarType(str, enum.Enum):
    MAIN = "主車"
    COME = "來車"

class Arrow(BaseModel):
    type: CarType
    start: Position   # 箭頭箭尖方向與轉向
    end: Position    # 箭頭箭尾所在方位
    direction: Direction


class Arrows(BaseModel):
    events: List[Arrow]

json_schema = Arrows.model_json_schema()

structured_outputs = StructuredOutputsParams(json=json_schema)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=2048,
    structured_outputs=structured_outputs
)

# custom_system_prompt = "你是一位專業的圖像分析助理，請用詳細步驟推理圖片內容，最後用中文回答。"

# question = """
# 假設地圖座標方位如下：
# - 箭頭往下為南，往上為北，往左傾斜為西，往右傾斜為東。
# 地圖上有兩組箭頭分別代表不同車輛：
# - 主車(即本路線由南向北直行箭頭)
# - 來車(即路口其他來向車流)

# 請依照箭頭尖端所指方向，逐一用繁體中文說明：
# 1. 主車 和 來車 從哪個方向(起點)出發，最終行駛到哪個方向(終點)。
# 2. 抽取主車、來車的：
#    - 車輛類型(如：主車、來車)
#    - 起始方向(以北/南/東/西表示)
#    - 終點方向(以北/南/東/西表示)
# """
# image_url = "images/collision_icons(阡陌)/側撞/北側路口遠端/右側撞_1.png"

# image_messages = [
#         {"role": "system", "content": custom_system_prompt},
#         {"role": "user", "content": [
#                 {"type": "text", "text": question},
#                 {
#                     "type": "image", 
#                     "image": image_url, 
#                 }
#             ]
#         },
#     ]

##############################################################
# --------------------Collision Description------------------# 
##############################################################

def safe_json_parse(json_str):
    """安全地解析 JSON 字串，處理常見格式問題"""
    try:
        # 移除可能的前後空白
        json_str = json_str.strip()
        
        # 如果字串被包在 ```json ``` 中，先移除這些標記
        if json_str.startswith('```json'):
            json_str = json_str.replace('```json', '').replace('```', '').strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失敗: {e}")
        return None

class RegionEnum(str, enum.Enum):
    # 中央
    路口中端 = "路口中端"

    # 西側
    路段_西_往西 = "路段-西-往西"
    路段_西_往東 = "路段-西-往東"
    鄰近路口_西 = "鄰近路口-西"
    路口近端_西 = "路口近端-西"
    路口遠端_西 = "路口遠端-西"

    # 東側
    路段_東_往東 = "路段-東-往東"
    路段_東_往西 = "路段-東-往西"
    鄰近路口_東 = "鄰近路口-東"
    路口近端_東 = "路口近端-東"
    路口遠端_東 = "路口遠端-東"

    # 南側
    路段_南_往南 = "路段-南-往南"
    路段_南_往北 = "路段-南-往北"
    鄰近路口_南 = "鄰近路口-南"
    路口近端_南 = "路口近端-南"
    路口遠端_南 = "路口遠端-南"

    # 北側
    路段_北_往北 = "路段-北-往北"
    路段_北_往南 = "路段-北-往南"
    鄰近路口_北 = "鄰近路口-北"
    路口近端_北 = "路口近端-北"
    路口遠端_北 = "路口遠端-北"

class CollisionEnum(str, enum.Enum):
    # 岔撞
    右轉岔撞 = "右轉岔撞"
    左轉岔撞 = "左轉岔撞"
    右岔撞 = "右岔撞"
    左岔撞 = "左岔撞"
    左前岔撞 = "左前岔撞"
    左後岔撞 = "左後岔撞"
    右前岔撞 = "右前岔撞"
    右後岔撞 = "右後岔撞"

    # 側撞
    右前側撞 = "右前側撞"
    右後側撞 = "右後側撞"
    右彎側撞 = "右彎側撞"
    右轉側撞 = "右轉側撞"
    右側撞 = "右側撞"
    右迴側撞 = "右迴側撞"
    左前側撞 = "左前側撞"
    左後側撞 = "左後側撞"
    左彎側撞 = "左彎側撞"
    左側撞 = "左側撞"
    左迴側撞 = "左迴側撞"

    # 追撞
    左彎追撞 = "左彎追撞"
    右彎追撞 = "右彎追撞"
    右車門追撞 = "右車門追撞"
    左車門追撞 = "左車門追撞"
    追撞 = "追撞"
    停等追撞 = "停等追撞"

    # 擦撞
    右匯入擦撞 = "右匯入擦撞"
    右對向擦撞 = "右對向擦撞"
    左匯入擦撞 = "左匯入擦撞"
    左對向擦撞 = "左對向擦撞"
    同右彎擦撞 = "同右彎擦撞"
    同左彎擦撞 = "同左彎擦撞"
    同向擦撞 = "同向擦撞"
    爭道擦撞 = "爭道擦撞"

    # 倒車撞
    右前倒車撞 = "右前倒車撞"
    右後倒車撞 = "右後倒車撞"
    右倒車撞 = "右倒車撞"
    左前倒車撞 = "左前倒車撞"
    左後倒車撞 = "左後倒車撞"
    左倒車撞 = "左倒車撞"
    前倒車撞 = "前倒車撞"
    倒車自撞 = "倒車自撞"
    倒車撞停車 = "倒車撞停車"

    # 迴轉撞
    右轉迴轉撞 = "右轉迴轉撞"
    左彎迴轉撞 = "左彎迴轉撞"

    # 碰撞
    雙右轉碰撞 = "雙右轉碰撞"
    雙左彎碰撞 = "雙左彎碰撞"

    # 對撞
    右彎對撞 = "右彎對撞"
    左彎對撞 = "左彎對撞"
    對撞 = "對撞"
    直行對撞 = "直行對撞"

    # 其他
    右彎自摔 = "右彎自摔"
    右彎自撞 = "右彎自撞"
    左彎自摔 = "左彎自摔"
    左彎自撞 = "左彎自撞"
    自摔 = "自摔"
    自撞 = "自撞"

# 定義結構
class Description(BaseModel):
    region: RegionEnum    # 用 enum 限制可填值
    collision: CollisionEnum

class Collision_Descroption(BaseModel):
    events: List[Description] 

# 生成 JSON Schema
json_descr_schema = Collision_Descroption.model_json_schema()

# Tool Calling 格式定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_collision_legend",
            "description": "分析多個事故撞擊區域、肇事類型",
            "parameters": json_descr_schema
        }
    }
]    

question = """
假設肇事型態分為如下：
岔撞（右轉岔撞、左轉岔撞、右岔撞、左岔撞、左前岔撞、左後岔撞、右前岔撞、右後岔撞）、
側撞（右前側撞、右後側撞、右彎側撞、右轉側撞、右側撞、右迴側撞、左前側撞、左後側撞、左彎側撞、左側撞、左迴側撞）、
追撞（左彎追撞、右彎追撞、右車門追撞、左車門追撞、追撞、停等追撞）、
擦撞（右匯入擦撞、右對向擦撞、左匯入擦撞、左對向擦撞、同右彎擦撞、同左彎擦撞、同向擦撞、爭道擦撞）、
倒車撞（右前倒車撞、右後倒車撞、右倒車撞、左前倒車撞、左後倒車撞、左倒車撞、前倒車撞、倒車自撞、倒車撞停車）、
迴轉撞（右轉迴轉撞、左彎迴轉撞）、
碰撞（雙右轉碰撞、雙左彎碰撞）、
對撞（右彎對撞、左彎對撞、對撞、直行對撞）、
其他（右彎自摔、右彎自撞、左彎自摔、左彎自撞、自摔、自撞）

易肇事區域分為如下：
路口中端、路段-西-往西、路段-西-往東、鄰近路口-西、路口近端-西、路口遠端-西、
路段-東-往東、路段-東-往西、鄰近路口-東、路口近端-東、路口遠端-東、
路段-南-往南、路段-南-往北、鄰近路口-南、路口近端-南、路口遠端-南、
路段-北-往北、路段-北-往南、鄰近路口-北、路口近端-北、路口遠端-北

地圖上有多組碰撞構圖圖例：
- 方形(即代表碰撞圖例的車門)
- 鄰近兩組箭頭(即主車及來車方向)
- 圓形中心(碰撞圖例的中心)

請基於提供的肇事碰撞構圖，分析地圖中多組碰撞構圖圖例的易肇事區域位置，並辨識各自的肇事型態：
1. 碰撞構圖圖例出現在地圖上的哪個區域
2. 該碰撞構圖圖例屬於哪一種肇事型態
"""

# # custom_system_prompt = "你是一位專業的圖像分析助理，請用詳細步驟推理圖片內容，最後使用中文回答。"
custom_system_prompt = "You are a helpful assistant that MUST respond with a JSON object following the schema"
