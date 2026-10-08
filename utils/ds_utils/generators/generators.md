# `utils/ds_utils/generators/` 子類別流程與差異說明

`utils/ds_utils/generators/` 底下的 6 個類別都繼承自 `utils/ds_utils/base.py` 的
`CollisionDiagramBase`，各自只實作一個 `generate_*` 方法，代表一個歷史上的資料格式迭代版本。
`utils/ds_utils/facade.py` 的 `CollisionDiagram` 用多重繼承把全部 6 個版本合併回同一個名稱，
供 `make_data.py`（`--generator` 參數）等既有程式碼使用；也可以直接從
`utils.ds_utils.generators` import 單一版本的子類別。

## 總覽比較表

| 類別 | 方法 | 支援來源 | 去重複邏輯 | 重疊偵測 | `offset_range` | `shrink_divisor` | render 函式 | 連續事件數疊圖 |
|---|---|---|---|---|---|---|---|---|
| `CollisionDescriptionV1` | `generate_v1` | 僅運研所 | 地點 + 主要型態 | 純量 IoU，門檻 0.01 | 15 | `FINAL_SHRINK_ICON`（4） | `add_icon_to_image` | 無 |
| `CollisionDescriptionV2` | `generate_v2` | 僅運研所 | 地點 + 主要型態 | 純量 IoU，門檻 0.01 | 15 | `FINAL_SHRINK_ICON`（4） | `add_icon_to_image` | 無 |
| `CollisionDescriptionV3` | `generate_v3` | 僅運研所 | 地點 + 主要型態 | 純量 IoU，門檻 0.01 | 5 | `FINAL_SHRINK_NUMBER_ICON`（6） | `add_number_icon_to_image` | 無（固定傳 `overlap=1`） |
| `CollisionDiagramMix` | `generate_mix` | 運研所／阡陌隨機 | 地點 + 主要型態 | 純量 IoU，門檻 0.01 | 5 | 依 `self.source` 動態決定 | 依 `self.source` 選 `add_number_icon_to_image` 或 `add_transport_icon_to_image` | 無（固定傳 `overlap=1`） |
| `CollisionDiagramTuneV1` | `generate_tune_v1` | 運研所／阡陌隨機 | 地點 + 主要型態 | 向量化 IoU，門檻 0.2 | 15 | 依 `self.source` 動態決定 | 同上 | 有（`select_overlap()`） |
| `CollisionDiagramTuneV2` | `generate_tune_v2` | 運研所／阡陌隨機 | **僅地點**（不去重複主要型態） | 向量化 IoU，門檻 0.2 | 15 | 依 `self.source` 動態決定 | 同上 | 有（`select_overlap()`） |

**JSONL 輸出欄位**：
- `CollisionDescriptionV1`/`V2`：`background`、`image`、`text`
- `CollisionDescriptionV3`：`background`、`image`、`text`
- `CollisionDiagramMix`：`source`、`background`、`image`、`text`
- `CollisionDiagramTuneV1`/`TuneV2`：`source`、`background`、`image`、`text`、`icons`、`road_type`、`ocr_nums`、`ocr_labels`

## 共用基底流程（`CollisionDiagramBase`）

所有 `generate_*` 方法開始前，driver 端（`make_data.py` 或測試腳本）都會先做這幾步，才呼叫
`generate_*`：

```mermaid
flowchart TD
    A([建立 CollisionDiagram 實例]) --> B["__init__ -> load_maps()：讀取 CSV 底圖清單"]
    B --> C["select_map()：隨機選一張底圖（僅第一次呼叫有效）"]
    C --> D["set_background()：設定 base_image 路徑與路口類型（正交/斜交/其他）"]
    D --> E(["呼叫對應的 generate_* 方法"])
```

`generate_*` 方法內部共用的每一輪迴圈邏輯，抽象後長這樣（各版本的差異點見下方個別流程圖）：

```mermaid
flowchart TD
    A([決定 self.source]) --> B["set_collision_types()：依 source 抽 collision_num 個碰撞型態"]
    B --> C{{逐一走訪 self.collisions}}
    C --> D["get_collision_locations()：依型態隨機取地點名稱 loc"]
    D --> E["get_thin_collision()：取細分類碰撞標籤"]
    E --> F[去重複檢查]
    F --> G{通過?}
    G -- 否 --> C
    G -- 是 --> H["location_to_position()：地點名稱轉座標 (x,y)"]
    H --> I{座標存在?}
    I -- 否 --> C
    I -- 是 --> J["select_icon_image() + get_los_parse_angle()：選圖示與旋轉角度"]
    J --> K["place_icon()：算貼圖座標與重疊判斷框"]
    K --> L[重疊偵測]
    L --> M{重疊?}
    M -- 是 --> C
    M -- 否 --> N[組資料 + 累積文字描述]
    N --> C
    C -- 走訪完畢 --> O{data 有內容?}
    O -- 是 --> P[呼叫對應 render 函式合成圖片並存檔]
```

---

## `CollisionDescriptionV1`（肇事型態描述 Ver.1）

最早期版本：只標註「在哪個地點發生什麼肇事類型」，不含行車方向、不含傷亡人數。固定使用運研所詞彙。

```mermaid
flowchart TD
    Start(["generate_v1(collision_num, save_path)"]) --> A["固定 self.source = 運研所"]
    A --> B[set_collision_types]
    B --> C{{逐一走訪 collisions}}
    C --> D[get_collision_locations -> loc]
    D --> E[get_thin_collision -> collision_label]
    E --> F{main_collision 已記錄?}
    F -- 是 --> C
    F -- 否 --> G{loc 已使用?}
    G -- 是 --> C
    G -- 否 --> H[location_to_position -> x, y]
    H --> I{x, y 為 None?}
    I -- 是 --> C
    I -- 否 --> J[select_icon_image + get_los_parse_angle]
    J --> K["place_icon(offset_range=15, shrink_divisor=4)"]
    K --> L["純量 IoU 比對 check_pos，門檻 0.01"]
    L --> M{任一 IoU > 0.01?}
    M -- 是 --> C
    M -- 否 --> N["文字: '在{loc}發生{main_collision}'"]
    N --> O["data.append((icon, pos, angle, location_mapping[loc], main_collision))"]
    O --> C
    C -- 完畢 --> P{data 有內容?}
    P -- 否 --> End([結束])
    P -- 是 --> Q[add_icon_to_image 合成圖片]
    Q --> R{有 save_path?}
    R -- 是 --> S["存 PNG + jsonl: background/image/text"]
    R -- 否 --> End
    S --> End
```

**重點**：資料 tuple 最簡單（5 欄，無 overlap 欄位），文字描述只有「在哪裡發生什麼型態」。

---

## `CollisionDescriptionV2`（肇事型態描述 Ver.2）

在 Ver.1 基礎上，多呼叫一次 `get_text_way_direction()` 加上行車方向描述；其餘流程與 Ver.1 完全相同。

```mermaid
flowchart TD
    Start(["generate_v2(collision_num, save_path)"]) --> A["固定 self.source = 運研所"]
    A --> B[set_collision_types]
    B --> C{{逐一走訪 collisions}}
    C --> D[get_collision_locations -> loc]
    D --> E[get_thin_collision -> collision_label]
    E --> F{main_collision 已記錄?}
    F -- 是 --> C
    F -- 否 --> G{loc 已使用?}
    G -- 是 --> C
    G -- 否 --> H[location_to_position -> x, y]
    H --> I{x, y 為 None?}
    I -- 是 --> C
    I -- 否 --> J[select_icon_image + get_los_parse_angle]
    J --> K["place_icon(offset_range=15, shrink_divisor=4)"]
    K --> L["純量 IoU 比對 check_pos，門檻 0.01"]
    L --> M{任一 IoU > 0.01?}
    M -- 是 --> C
    M -- 否 --> N2["get_text_way_direction -> way_direction（Ver.1 沒有這步）"]
    N2 --> N["文字: '{way_direction}，兩車在{loc}發生{main_collision}'"]
    N --> O["data.append((icon, pos, angle, location_mapping[loc], main_collision))"]
    O --> C
    C -- 完畢 --> P{data 有內容?}
    P -- 否 --> End([結束])
    P -- 是 --> Q[add_icon_to_image 合成圖片]
    Q --> R{有 save_path?}
    R -- 是 --> S["存 PNG + jsonl: background/image/text"]
    R -- 否 --> End
    S --> End
```

**與 Ver.1 的唯一差異**：多了 `get_text_way_direction()` 這一步，文字描述多了行車方向。資料
tuple 結構與 render 函式（`add_icon_to_image`）跟 Ver.1 相同。

---

## `CollisionDescriptionV3`（肇事型態描述 Ver.3）

在 Ver.2 基礎上再加上傷亡人數：每張圖大約一半的圖示會標註受傷人數（`ocr_nums < collision_num/2`
時才給非零人數，其餘固定 0）。改用 `add_number_icon_to_image` 繪製（會在圖示上畫數字），因此
`shrink_divisor`/`offset_range` 都跟 Ver.1/2 不同。

```mermaid
flowchart TD
    Start(["generate_v3(collision_num, save_path)"]) --> A["固定 self.source = 運研所"]
    A --> B[set_collision_types]
    B --> C{{逐一走訪 collisions}}
    C --> D[get_collision_locations -> loc]
    D --> E[get_thin_collision -> collision_label]
    E --> F{main_collision 已記錄?}
    F -- 是 --> C
    F -- 否 --> G{loc 已使用?}
    G -- 是 --> C
    G -- 否 --> H[location_to_position -> x, y]
    H --> I{x, y 為 None?}
    I -- 是 --> C
    I -- 否 --> J[select_icon_image + get_los_parse_angle]
    J --> K["place_icon(offset_range=5, shrink_divisor=6)"]
    K --> L["純量 IoU 比對 check_pos，門檻 0.01"]
    L --> M{任一 IoU > 0.01?}
    M -- 是 --> C
    M -- 否 --> N2[get_text_way_direction -> way_direction]
    N2 --> N3{"ocr_nums < collision_num/2 ?"}
    N3 -- 是 --> N4["get_random_hurt() -> 隨機 1~14 人，ocr_nums+1"]
    N3 -- 否 --> N5["get_random_hurt(force_0=True) -> 固定 0 人"]
    N4 --> N6["文字: '{hurt_info}{way_direction}，兩車在{loc}發生{main_collision}'"]
    N5 --> N6
    N6 --> O["data.append((1, icon, (0,0), pos, angle, hurt_num, location_mapping[loc], main_collision, collision_label, way_direction, orign_description))"]
    O --> C
    C -- 完畢 --> P{data 有內容?}
    P -- 否 --> End([結束])
    P -- 是 --> Q["add_number_icon_to_image(data, debug=False)"]
    Q --> R{有 save_path?}
    R -- 是 --> S["存 PNG + jsonl: background/image/text"]
    R -- 否 --> End
    S --> End
```

**重點**：資料 tuple 開頭固定塞 `1`（overlap 欄位，代表不疊貼——這個版本沒有連續事件數的
概念，純粹是為了配合 `add_number_icon_to_image` 統一後的資料格式）。JSONL 輸出欄位跟
Ver.1/2 一樣簡單（沒有 `source`/`icons`/`road_type` 等欄位），因為固定只用運研所。

---

## `CollisionDiagramMix`（運研所／阡陌混合來源）

第一個支援「阡陌」詞彙的版本。整張圖開始前用 `get_source()` 隨機決定 `self.source`，之後兩種
來源在迴圈中段分岔：阡陌分支比照 Ver.3 加傷亡人數、運研所分支比照 Ver.2 只加行車方向。

```mermaid
flowchart TD
    Start(["generate_mix(collision_num, save_path)"]) --> A["get_source() -> self.source 隨機 運研所/阡陌"]
    A --> B[set_collision_types]
    B --> C{{逐一走訪 collisions}}
    C --> D[get_collision_locations -> loc]
    D --> E[get_thin_collision -> collision_label]
    E --> F{main_collision 已記錄?}
    F -- 是 --> C
    F -- 否 --> G{loc 已使用?}
    G -- 是 --> C
    G -- 否 --> H[location_to_position -> x, y]
    H --> I{x, y 為 None?}
    I -- 是 --> C
    I -- 否 --> J[select_icon_image + get_los_parse_angle]
    J --> K{"self.source == 阡陌 ?"}
    K -- 是 --> K1["shrink_divisor = FINAL_SHRINK_NUMBER_ICON(6)"]
    K -- 否 --> K2["shrink_divisor = FINAL_SHRINK_TRANSPORT_ICON(3)"]
    K1 --> L["place_icon(offset_range=5, shrink_divisor)"]
    K2 --> L
    L --> M["純量 IoU 比對 check_pos，門檻 0.01"]
    M --> N{任一 IoU > 0.01?}
    N -- 是 --> C
    N -- 否 --> O[get_text_way_direction -> way_direction]
    O --> P{"self.source == 阡陌 ?"}
    P -- 是 --> Q1[依 ocr_nums 決定受傷人數]
    Q1 --> Q2["data.append((1, icon, (0,0), pos, angle, hurt_num, inverse_location_mapping[loc], ...))"]
    P -- 否 --> Q3["data.append((1, icon, pos, angle, location_mapping[loc], main_collision))"]
    Q2 --> C
    Q3 --> C
    C -- 完畢 --> R{data 有內容?}
    R -- 否 --> End([結束])
    R -- 是 --> S{"self.source == 阡陌 ?"}
    S -- 是 --> S1[add_number_icon_to_image]
    S -- 否 --> S2[add_transport_icon_to_image]
    S1 --> T{有 save_path?}
    S2 --> T
    T -- 是 --> U["存 PNG + jsonl: source/background/image/text"]
    T -- 否 --> End
    U --> End
```

**重點**：`self.source` 在**整張圖生成前只決定一次**（不是每個碰撞各自決定），所以同一張圖
裡的所有圖示都是同一套詞彙。`shrink_divisor`/render 函式依 source 分岔，但兩者的資料 tuple
都固定帶 `overlap=1`（這個版本還沒有連續事件數功能）。JSONL 多了 `source` 欄位。

---

## `CollisionDiagramTuneV1`

在 `Mix` 基礎上，把重疊偵測從「純量逐一比對、門檻 0.01」換成「向量化 IoU、門檻 0.2」（效能較好，
且門檻放寬到 0.2，允許一定程度的框重疊仍算「沒有衝突」），並新增「連續事件數」疊圖功能
（`select_overlap()`：同一個地點的圖示有機率沿對角線疊貼 6~9 次，代表多起連續事故）。
**跟 `TuneV2` 唯一的差異**是這個版本仍保留「主要型態去重複」（`record_main_collisions`）。

```mermaid
flowchart TD
    Start(["generate_tune_v1(collision_num, save_path)"]) --> A["get_source() -> self.source 隨機"]
    A --> B["set_collision_types; self.over_overlap = 0"]
    B --> C{{逐一走訪 collisions}}
    C --> D[get_collision_locations -> loc]
    D --> E[get_thin_collision -> collision_label]
    E --> F{"main_collision 已記錄? (與 TuneV2 的差異點)"}
    F -- 是 --> C
    F -- 否 --> G{loc 已使用?}
    G -- 是 --> C
    G -- 否 --> H[location_to_position -> x, y]
    H --> I{x, y 為 None?}
    I -- 是 --> C
    I -- 否 --> J[select_icon_image + get_los_parse_angle]
    J --> K{"self.source == 阡陌 ?"}
    K -- 是 --> K1["shrink_divisor = 6"]
    K -- 否 --> K2["shrink_divisor = 3"]
    K1 --> L["place_icon(offset_range=15, shrink_divisor)"]
    K2 --> L
    L --> M["compute_iou_vectorized(curr_box, check_pos)，門檻 0.2"]
    M --> N{任一 IoU > 0.2?}
    N -- 是 --> Skip[skip_coll.append] --> C
    N -- 否 --> O[get_text_way_direction -> way_direction]
    O --> P{"self.source == 阡陌 ?"}
    P -- 是 --> Q1[依 ocr_nums 決定受傷人數]
    Q1 --> Q2["select_overlap() -> overlap 改為連續事件數 1 或 6~9"]
    Q2 --> Q3["data.append((overlap, icon, (0,0), pos, angle, hurt_num, inverse_location_mapping[loc], ...))"]
    P -- 否 --> R1["select_overlap() -> overlap 改為連續事件數"]
    R1 --> R2["data.append((overlap, icon, pos, angle, location_mapping[loc], main_collision))"]
    Q3 --> C
    R2 --> C
    C -- 完畢 --> S["印出: 碰撞數量 / 去除重疊圖例數 (len skip_coll)"]
    S --> T{data 有內容?}
    T -- 否 --> End([結束, 回傳 ix-1])
    T -- 是 --> U{"self.source == 阡陌 ?"}
    U -- 是 --> U1[add_number_icon_to_image]
    U -- 否 --> U2[add_transport_icon_to_image]
    U1 --> V{有 save_path?}
    U2 --> V
    V -- 是 --> W["存 PNG + jsonl: source/background/image/text/icons/road_type/ocr_nums/ocr_labels"]
    V -- 否 --> End
    W --> End
```

**重點**：
- 重疊偵測改用 `compute_iou_vectorized`，一次跟 `check_pos` 裡所有框比較（效能較好），且門檻
  放寬到 0.2。
- `select_overlap()` 把原本代表「是否重疊」的 `overlap` 變數重新賦值成「連續事件數」
  （1 表示不疊貼，6~9 表示要疊貼幾次），交給 render 函式畫出對角線疊貼效果。
- 仍保留 `record_main_collisions` 去重複（跟 `Description*`/`Mix` 一致）。
- JSONL 多了 `icons`（實際成功放置的圖示數）、`road_type`、`ocr_nums`、`ocr_labels`。
- 方法有回傳值 `ix - 1`（實際放置數量），其餘版本都沒有回傳值。

---

## `CollisionDiagramTuneV2`（目前主力版本，`make_data.py` 預設使用）

流程與 `TuneV1` **完全相同**，唯一差異是迴圈裡**沒有** `main_collision 已記錄過就跳過` 這個
檢查，所以同一張圖可以出現同一種主要肇事型態多次——這是刻意拿掉去重複限制以增加資料多樣性，
不是遺漏。

```mermaid
flowchart TD
    Start(["generate_tune_v2(collision_num, save_path)"]) --> A["get_source() -> self.source 隨機"]
    A --> B["set_collision_types; self.over_overlap = 0"]
    B --> C{{逐一走訪 collisions}}
    C --> D[get_collision_locations -> loc]
    D --> E["get_thin_collision -> collision_label（不檢查 main_collision 是否重複，與 TuneV1 的差異點）"]
    E --> G{loc 已使用?}
    G -- 是 --> C
    G -- 否 --> H[location_to_position -> x, y]
    H --> I{x, y 為 None?}
    I -- 是 --> C
    I -- 否 --> J[select_icon_image + get_los_parse_angle]
    J --> K{"self.source == 阡陌 ?"}
    K -- 是 --> K1["shrink_divisor = 6"]
    K -- 否 --> K2["shrink_divisor = 3"]
    K1 --> L["place_icon(offset_range=15, shrink_divisor)"]
    K2 --> L
    L --> M["compute_iou_vectorized(curr_box, check_pos)，門檻 0.2"]
    M --> N{任一 IoU > 0.2?}
    N -- 是 --> Skip[skip_coll.append] --> C
    N -- 否 --> O[get_text_way_direction -> way_direction]
    O --> P{"self.source == 阡陌 ?"}
    P -- 是 --> Q1[依 ocr_nums 決定受傷人數]
    Q1 --> Q2["select_overlap() -> overlap 改為連續事件數 1 或 6~9"]
    Q2 --> Q3["data.append((overlap, icon, (0,0), pos, angle, hurt_num, inverse_location_mapping[loc], ...))"]
    P -- 否 --> R1["select_overlap() -> overlap 改為連續事件數"]
    R1 --> R2["data.append((overlap, icon, pos, angle, location_mapping[loc], main_collision))"]
    Q3 --> C
    R2 --> C
    C -- 完畢 --> S["印出: 碰撞數量 / 去除重疊圖例數 (len skip_coll)"]
    S --> T{data 有內容?}
    T -- 否 --> End([結束, 回傳 ix-1])
    T -- 是 --> U{"self.source == 阡陌 ?"}
    U -- 是 --> U1[add_number_icon_to_image]
    U -- 否 --> U2[add_transport_icon_to_image]
    U1 --> V{有 save_path?}
    U2 --> V
    V -- 是 --> W["存 PNG + jsonl: source/background/image/text/icons/road_type/ocr_nums/ocr_labels"]
    V -- 否 --> End
    W --> End
```

**與 `TuneV1` 的唯一差異**：拿掉了 `main_collision` 去重複檢查（上圖的 E 步驟），其餘完全相同
（重疊偵測、連續事件數疊圖、JSONL 欄位、回傳值都一樣）。

---

## 版本演進脈絡（由舊到新）

```mermaid
flowchart LR
    V1["CollisionDescriptionV1<br/>地點+型態"] --> V2["CollisionDescriptionV2<br/>+行車方向"]
    V2 --> V3["CollisionDescriptionV3<br/>+傷亡人數"]
    V3 --> Mix["CollisionDiagramMix<br/>+運研所/阡陌混合來源"]
    Mix --> TuneV1["CollisionDiagramTuneV1<br/>+向量化重疊偵測+連續事件數"]
    TuneV1 --> TuneV2["CollisionDiagramTuneV2（目前主力）<br/>拿掉主要型態去重複"]
```

## 已知問題與注意事項

- `CollisionDescriptionV1`/`V2`/`V3` 只支援運研所詞彙（`self.source` 在方法內固定寫死），
  呼叫端不需要（也不應該）自己設定 `self.source`。
- `place_icon()` 的 `shrink_divisor` 參數**必須**跟該版本最終會呼叫的 render 函式一致，兩者
  對不上會導致重疊偵測失真（2026-07-24 修過的既有 bug，詳見 `markdown/NEED.md`）。
- `CollisionDiagramMix`/`TuneV1`/`TuneV2` 的 `self.source` 是「整張圖決定一次」，不是每個
  碰撞各自隨機——同一張圖裡的圖示一定同屬一套詞彙。
- 若要驗證任一版本改動後是否還能正常運作，用 `python make_test_data.py` 逐一測試 6 個版本，
  不要只測自己改的那一個（`CollisionDiagramBase` 的共用方法一改，全部版本都會受影響）。
