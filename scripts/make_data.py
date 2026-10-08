
import argparse
import json
import os
import shutil
import sys
import time

# 這支腳本移到 scripts/ 底下後，Python 預設只會把 scripts/ 本身加進 sys.path，
# 找不到專案根目錄的 utils 套件；這裡手動把根目錄加回 sys.path，不論從哪個
# CWD 呼叫 `python scripts/make_data.py` 都能正確 import。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Features, Image, Value, load_dataset, load_from_disk

from utils.ds_utils import (
    CollisionDescriptionV1,
    CollisionDescriptionV2,
    CollisionDescriptionV3,
    CollisionDiagramMix,
    CollisionDiagramTuneV1,
    CollisionDiagramTuneV2,
    pipeline,
)
from utils.ds_utils.io_utils import check_image_record, is_valid_record

user_prompt = "你是一位專精台灣交通工程與交通安全的專家，請用繁體中文回答我的專業問題。請基於提供的肇事碰撞構圖，對路口進行分析並辨識肇事型態。"

# 每個資料格式版本對應的 CollisionDiagram 子類別，與其 generate 方法名稱（見 utils/ds_utils/generators/）
GENERATOR_REGISTRY = {
    "CollisionDescriptionV1": (CollisionDescriptionV1, "generate_v1"),
    "CollisionDescriptionV2": (CollisionDescriptionV2, "generate_v2"),
    "CollisionDescriptionV3": (CollisionDescriptionV3, "generate_v3"),
    "CollisionDiagramMix": (CollisionDiagramMix, "generate_mix"),
    "CollisionDiagramTuneV1": (CollisionDiagramTuneV1, "generate_tune_v1"),
    "CollisionDiagramTuneV2": (CollisionDiagramTuneV2, "generate_tune_v2"),
}

# 各版本的簡短說明，顯示在 --generator 的 --help 裡
GENERATOR_DESCRIPTIONS = {
    "CollisionDescriptionV1": "肇事型態描述 Ver.1，只有路段與肇事類型（不含行車方向、不含傷亡人數）。"
                              "只支援運研所圖示來源。",
    "CollisionDescriptionV2": "肇事型態描述 Ver.2，在 Ver.1 基礎上加上行車方向描述。"
                              "只支援運研所圖示來源。",
    "CollisionDescriptionV3": "肇事型態描述 Ver.3，在 Ver.2 基礎上加上傷亡人數（約一半的圖示會標註人數）。"
                              "只支援運研所圖示來源。",
    "CollisionDiagramMix": "每張圖隨機混合運研所／阡陌兩種圖示來源與詞彙，運研所分支含行車方向、"
                           "阡陌分支含傷亡人數。",
    "CollisionDiagramTuneV1": "在 Mix 基礎上改用向量化重疊偵測與連續事件數描述；同一張圖裡，"
                              "同一種主要肇事型態只會出現一次（跨型態去重複）。",
    "CollisionDiagramTuneV2": "目前主力版本，與 TuneV1 唯一差異是拿掉了跨型態去重複的限制，"
                              "同一張圖可以出現同一種主要肇事型態多次，換取更高的資料多樣性。",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="合成交通事故碰撞構圖訓練資料",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    generator_help = "要使用哪個 CollisionDiagram 子類別（資料格式版本）合成資料，預設 CollisionDiagramTuneV2（目前主力版本）：\n" + "\n".join(
        f"  {name}: {GENERATOR_DESCRIPTIONS[name]}" for name in GENERATOR_REGISTRY
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=list(GENERATOR_REGISTRY.keys()),
        default="CollisionDiagramTuneV2",
        help=generator_help,
    )
    parser.add_argument(
        "--dataset_tag",
        type=str,
        default="tune_v2",
        help=(
            "資料集版本標籤，同時決定輸出目錄與 pickle 快取檔名（兩者原本只差在名字跟目錄"
            "路徑，這裡統一成一個參數）：\n"
            "  輸出目錄 -> data/collision_diagram/synthetic_<dataset_tag>/\n"
            "  pickle 快取 -> data/train_dataset_<dataset_tag>.pkl 與 data/eval_dataset_<dataset_tag>.pkl"
        ),
    )
    parser.add_argument("--train_num", type=int, default=30000, help="train split 要合成的圖片數量")
    parser.add_argument("--val_num", type=int, default=100, help="val split 要合成的圖片數量")
    parser.add_argument("--test_num", type=int, default=100, help="test split 要合成的圖片數量")
    parser.add_argument("--rand_num_min", type=int, default=30, help="每張圖隨機產生的碰撞圖例數量下限（可能重疊）")
    parser.add_argument("--rand_num_max", type=int, default=40, help="每張圖隨機產生的碰撞圖例數量上限（可能重疊）")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["build", "check"],
        default="build",
        help=(
            "build: 依原本流程合成資料、格式化並快取成 pickle（預設，行為與原本相同）。\n"
            "check: 不合成任何新資料，只檢查 --dataset_tag 目前已建立好的內容是否完整：\n"
            "  - 每個 split 的 jsonl 行數、background/text 是否有空值\n"
            "  - jsonl 裡的 image 路徑是否存在、圖片能否正常開啟（偵測缺檔/損毀圖片）\n"
            "  - data/{train,eval}_dataset_<dataset_tag>.pkl 是否存在、能否正常讀取\n"
            "  只印報告，不會刪除或修改任何檔案。"
        ),
    )
    return parser.parse_args()

# Convert dataset to OAI messages。
#
# 2026-09-02 修正：content 底下每一項原本依 type 只帶自己需要的欄位（text 項只有
# "text"，image 項只有 "image"），造成同一個 list 裡的 dict 形狀不一致。pyarrow 沒辦法把
# 這種不一致的形狀推斷成單一 struct 型別，只能整欄位退回用 ujson_dumps 做 JSON 字串序列化
# ——但 image 項此時已經是解碼後的原始圖片 bytes，不是合法 UTF-8 文字，一併塞進
# ujson_dumps 就會炸 OverflowError（在 build_dataset() 呼叫 .map(format_data, ...) 時
# 於 arrow_writer.py 內部發生）。修法是讓每一項固定同時帶 text/image 兩個欄位（用不到的
# 設 None），並透過下面的 MESSAGE_FEATURES 明確宣告輸出 schema，讓 image 欄位維持用
# datasets.Image() 原生嵌入，不需要經過 JSON 序列化這條路徑。
def format_data(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                        "image": None,
                    },
                    {
                        "type": "image",
                        "text": None,
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"], "image": None}],
            },
        ],
    }


# format_data() 的輸出 schema：content 底下每一項固定有 type/text/image 三個欄位
# （text 或 image 用不到時為 None），image 欄位用 datasets.Image() 原生型別。
MESSAGE_FEATURES = Features({
    "messages": [
        {
            "role": Value("string"),
            "content": [
                {
                    "type": Value("string"),
                    "text": Value("string"),
                    "image": Image(),
                }
            ],
        }
    ]
})

def build_dataset(args):
    generator_cls, generate_method_name = GENERATOR_REGISTRY[args.generator]
    print(f"[make_data] 使用生成器: {generator_cls.__name__}.{generate_method_name}()")

    set_num = {
        "train": args.train_num,
        "val": args.val_num,
        "test": args.test_num,
    }

    # data_path 跟 pickle_file 原本是兩個獨立參數，但只差在名字跟目錄路徑（都是同一個
    # dataset_tag 加上不同的前綴），統一從 --dataset_tag 推導，避免兩者傳的值對不上。
    data_path = f"data/collision_diagram/synthetic_{args.dataset_tag}"
    pickle_file = f"dataset_{args.dataset_tag}"

    # ========================================================================
    # 1) 合成碰撞構圖資料：對 train/val/test 三個 split，交給 utils/ds_utils/pipeline.py
    #    的 run()。單筆樣本合成失敗（IconAssetNotFoundError）時會被記錄並跳過、繼續合成
    #    下一筆，不會讓整個 run 中斷（詳見 pipeline.py docstring）。
    # ========================================================================
    pipeline.run(
        generator_cls=generator_cls,
        generate_method_name=generate_method_name,
        set_num=set_num,
        data_path=data_path,
        rand_num_min=args.rand_num_min,
        rand_num_max=args.rand_num_max,
    )

    # ========================================================================
    # 2) 讀取上一步產生的 jsonl，組成 HuggingFace Dataset：過濾掉沒有底圖（background 為空
    #    字串）或圖片缺檔/損毀的髒資料（is_valid_record()，跟 check_dataset() 共用同一份
    #    驗證邏輯），並把 image 欄位轉成可載入的 datasets.Image 型別，供下一步轉成 OAI
    #    messages 格式使用。這裡先濾除壞掉的圖片，是為了避免下面的 .map(format_data, ...)
    #    才在讀圖片時因為某張圖壞掉而整批中斷（HuggingFace datasets.map() 沒有部分跳過
    #    機制）。
    # ========================================================================
    train_dataset, eval_dataset, test_dataset = load_dataset("json",
                        data_files={
                                        "train": [
                                            f"./{data_path}/train/train.jsonl",
                                        ],
                                        "val": [
                                            f"./{data_path}/val/val.jsonl",
                                        ],
                                        "test": [
                                            f"./{data_path}/test/test.jsonl",
                                        ],
                                }
                        , split= ["train", "val", "test"]
                    )
    print(train_dataset)
    train_dataset = train_dataset.filter(is_valid_record)
    eval_dataset = eval_dataset.filter(is_valid_record)
    print(eval_dataset)
    train_dataset = train_dataset.cast_column("image", Image())
    eval_dataset = eval_dataset.cast_column("image", Image())

    # ========================================================================
    # 3) 轉成 OAI messages 格式並用 datasets.save_to_disk() 快取（Arrow 格式，目錄底下含
    #    多個 shard 檔）：若 data/{train,eval}_{pickle_file}.pkl 兩個目錄都存在就直接用
    #    load_from_disk() 讀取（跳過整個 format_data 轉換，詳見 CLAUDE.md「若要強制重新
    #    產生需先刪除 data/ 底下對應的 .pkl」），否則才呼叫 format_data() 逐筆轉換並寫入
    #    新的快取。讀取時保留原始（過濾/cast 完成的）HF Dataset 在 raw_*_dataset，這樣即使
    #    快取損毀/寫到一半中斷（load_from_disk() 拋例外）也能安全地退回重新產生，而不會拿到
    #    一半是快取、一半是 HF Dataset 的髒狀態。
    #
    #    注意：儘管變數/檔名沿用「pkl」，save_to_disk() 實際寫出的是「目錄」而非單一檔案；
    #    讀取跟清除快取都要用對應目錄的 API（load_from_disk() / shutil.rmtree()），不能用
    #    joblib.load() / os.remove()（那兩個是給單一檔案用的，對目錄會直接拋例外）。
    # ========================================================================
    raw_train_dataset, raw_eval_dataset = train_dataset, eval_dataset
    train_pkl_path = f"data/train_{pickle_file}.pkl"
    eval_pkl_path = f"data/eval_{pickle_file}.pkl"
    cache_ready = os.path.exists(train_pkl_path) and os.path.exists(eval_pkl_path)

    if cache_ready:
        print("✅ 開始讀取快取...")
        start_time = time.time()
        try:
            train_dataset = load_from_disk(train_pkl_path)
            eval_dataset = load_from_disk(eval_pkl_path)
            elapsed_time = time.time() - start_time
            print(f"✅ 成功讀取快取！耗時 {elapsed_time:.2f} 秒")
        except Exception as e:
            # 快取損毀或寫到一半就被中斷（例如上次執行途中被 kill／OOM），
            # 不要整支腳本直接崩潰：刪掉壞掉的快取目錄，回退到下面重新產生。
            print(f"⚠️ 讀取快取失敗（{type(e).__name__}: {e}），視為快取損毀，刪除後重新產生...")
            for p in (train_pkl_path, eval_pkl_path):
                if os.path.exists(p):
                    shutil.rmtree(p)
            # 兩個 load_from_disk() 只要有一個失敗就整組回退：把 train_dataset/eval_dataset
            # 重設回過濾/cast 完成、尚未 format_data 的 raw_*_dataset，避免出現「train 已經是
            # 快取讀出來的格式化結果、eval 卻失敗」的髒狀態（否則下面 map() 的
            # remove_columns=train_dataset.column_names 會把 messages 欄位也删掉）。
            train_dataset, eval_dataset = raw_train_dataset, raw_eval_dataset
            cache_ready = False

    if not cache_ready:
        os.makedirs("data", exist_ok=True)
        # Convert dataset to OAI messages
        # map() 逐筆（或逐 batch）轉換，Arrow 底層會分批寫檔，不會把全部
        # 資料同時攤在記憶體裡；num_proc 可視 CPU 核數調整平行度。
        train_dataset = train_dataset.map(
            format_data,
            remove_columns=train_dataset.column_names,
            features=MESSAGE_FEATURES,  # 明確宣告 schema，避免 image 欄位退回 JSON 序列化
            desc="格式化 train 為 OAI messages",
            num_proc=4,          # 依機器 CPU 數量調整，記憶體吃緊可調小或拿掉
            writer_batch_size=500,  # 每 500 筆 flush 一次到磁碟，避免累積在記憶體
        )
        eval_dataset = eval_dataset.map(
            format_data,
            remove_columns=eval_dataset.column_names,
            features=MESSAGE_FEATURES,
            desc="格式化 eval 為 OAI messages",
            num_proc=4,
            writer_batch_size=500,
        )

        train_dataset.save_to_disk(train_pkl_path)
        eval_dataset.save_to_disk(eval_pkl_path)
        print("✅ 使用 Arrow (save_to_disk) 儲存完成！")

    if train_dataset:
        sample_idx = min(345, len(train_dataset) - 1)
        print(train_dataset[sample_idx]["messages"])


def check_dataset(args):
    """檢查 --dataset_tag 已建立好的 jsonl/圖片/pickle 是否完整，唯讀，不修改任何檔案。"""
    data_path = f"data/collision_diagram/synthetic_{args.dataset_tag}"
    pickle_file = f"dataset_{args.dataset_tag}"

    print(f"[make_data] 檢查模式，dataset_tag={args.dataset_tag}")
    print(f"  圖片/jsonl 目錄: {data_path}")
    print(f"  pickle 快取: data/train_{pickle_file}.pkl, data/eval_{pickle_file}.pkl")

    any_problem = False

    for set_name in ["train", "val", "test"]:
        jsonl_path = f"{data_path}/{set_name}/{set_name}.jsonl"
        print(f"\n--- {set_name}: {jsonl_path} ---")
        if not os.path.exists(jsonl_path):
            print("  找不到 jsonl 檔")
            any_problem = True
            continue

        total = 0
        empty_background = 0
        empty_text = 0
        missing_images = []
        corrupt_images = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                record = json.loads(line)

                if not record.get("background"):
                    empty_background += 1
                if not record.get("text"):
                    empty_text += 1

                ok, reason = check_image_record(record)
                if not ok:
                    if reason == "missing":
                        missing_images.append(record.get("image", ""))
                    else:
                        corrupt_images.append((record.get("image", ""), reason))

        print(f"  jsonl 筆數: {total}")
        print(f"  background 為空: {empty_background}")
        print(f"  text 為空: {empty_text}")
        print(f"  圖片缺檔: {len(missing_images)}")
        for p in missing_images[:5]:
            print(f"    - {p}")
        if len(missing_images) > 5:
            print(f"    ...（還有 {len(missing_images) - 5} 筆）")
        print(f"  圖片損毀/無法開啟: {len(corrupt_images)}")
        for p, err in corrupt_images[:5]:
            print(f"    - {p}: {err}")
        if len(corrupt_images) > 5:
            print(f"    ...（還有 {len(corrupt_images) - 5} 筆）")

        split_ok = total > 0 and not missing_images and not corrupt_images and empty_background == 0
        print(f"  結論: {'✅ 正常' if split_ok else '⚠️ 有問題'}")
        any_problem = any_problem or not split_ok

    print("\n--- pickle 快取 ---")
    for split_label, pkl_path in (
        ("train", f"data/train_{pickle_file}.pkl"),
        ("eval", f"data/eval_{pickle_file}.pkl"),
    ):
        if not os.path.exists(pkl_path):
            print(f"  {split_label}: 找不到 {pkl_path}")
            any_problem = True
            continue

        try:
            start_time = time.time()
            data = load_from_disk(pkl_path)
            elapsed = time.time() - start_time
            print(f"  {split_label}: ✅ {pkl_path}（讀取成功，共 {len(data)} 筆，耗時 {elapsed:.2f}s）")
        except Exception as e:
            print(f"  {split_label}: ⚠️ {pkl_path}（讀取失敗：{type(e).__name__}: {e}）")
            any_problem = True

    print(f"\n[make_data] 檢查完成，{'發現問題，請確認上方細節' if any_problem else '一切正常'}。")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "check":
        check_dataset(args)
    else:
        build_dataset(args)
