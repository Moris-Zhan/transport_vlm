"""原始樣本 -> OAI messages 格式轉換，以及訓練前的 image placeholder token 對齊檢查。"""

DEFAULT_IMAGE_TOKEN_IDS = [10, 12, 13]


def format_data(sample: dict, user_prompt: str) -> dict:
    """把一筆原始資料（image/text）轉成 OAI 格式的 messages。"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ],
    }


def process_vision_info(messages: list) -> list:
    """從 messages 中取出所有圖片內容，轉成 RGB PIL Image 列表。"""
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                image = element["image"] if "image" in element else element
                image_inputs.append(image.convert("RGB"))
    return image_inputs


def resolve_image_token_ids(processor) -> list:
    """從 processor 讀取 image_ids（可能因版本不同而沒有），沒有時退回預設值
    DEFAULT_IMAGE_TOKEN_IDS，並在讀到的值跟預設值不同時印出警告 —— collate_fn 的 labels
    遮罩用的是硬編碼 DEFAULT_IMAGE_TOKEN_IDS，若兩邊悄悄不一致，遮罩可能算錯。
    """
    image_token_ids = list(getattr(processor, "image_ids", None) or DEFAULT_IMAGE_TOKEN_IDS)
    if image_token_ids != DEFAULT_IMAGE_TOKEN_IDS:
        print(
            f"⚠️ processor.image_ids={image_token_ids} 與 collate_fn 硬編碼的 "
            f"{DEFAULT_IMAGE_TOKEN_IDS} 不一致，labels 的 image token 遮罩可能算錯，請確認。"
        )
    return image_token_ids


def assert_image_token_alignment(
    batch: dict,
    examples: list,
    patch_size: int,
    spatial_merge_size: int,
    image_token_ids: list,
) -> None:
    """驗證 batch 內每筆樣本的 image placeholder token 數，是否與該圖片 resize 後
    尺寸依 patch_size 換算出的理論值一致。假設每筆樣本恰好一張圖（format_data 的行為）。

    追查 Segmentation fault（訓練跑到 step ~616 時無 traceback 直接死掉）的頭號嫌疑：手動
    把 processor.patch_size / spatial_merge_size 覆寫成模型實際的值。若這個覆寫跟模型 vision
    tower 實際吃的 patch 佈局有任何落差，理論上 transformers 會在 modeling_mistral3.py 的
    masked_scatter 前用 `raise ValueError("Image features and image tokens do not match...")`
    攔下來（有清楚 traceback），不該是無聲的裸 segfault——但這裡仍在送進模型 forward 之前，於
    DataLoader worker 內先獨立驗證一次，公式與
    transformers/models/pixtral/processing_pixtral.py 的 PixtralProcessor.__call__ 完全一致，
    好在真正出事之前用更精確的錯誤訊息（哪個 batch、哪張圖）攔下來。
    """
    if "image_sizes" not in batch:
        return  # 這個 batch 沒有圖片，略過

    effective_patch = patch_size * spatial_merge_size
    input_ids = batch["input_ids"]
    image_sizes = batch["image_sizes"]

    for i, example in enumerate(examples):
        height, width = (int(v) for v in image_sizes[i])
        # 與 PixtralProcessor.__call__ 完全相同的公式：每列 num_width_tokens 個
        # [IMG] + 1 個 [IMG_BREAK]/[IMG_END]，共 num_height_tokens 列。
        num_height_tokens = height // effective_patch
        num_width_tokens = width // effective_patch
        expected_tokens = num_height_tokens * (num_width_tokens + 1)

        row = input_ids[i]
        actual_tokens = sum(int((row == tok_id).sum().item()) for tok_id in image_token_ids)

        if actual_tokens != expected_tokens:
            image_obj = example.get("image")
            image_path = getattr(image_obj, "filename", None)
            raise AssertionError(
                "image token 數量與圖片尺寸換算結果不一致（懷疑的 Segmentation fault 成因①）：\n"
                f"  batch 內索引={i}, image_path={image_path}, background={example.get('background')}\n"
                f"  resize 後尺寸(H,W)=({height},{width}), effective_patch={effective_patch}\n"
                f"  預期 token 數={expected_tokens}, 實際 token 數={actual_tokens}"
            )


def assert_image_token_alignment_glm(
    batch: dict,
    examples: list,
    merge_size: int,
    image_token_ids: list,
) -> None:
    """GLM-4.1V（Glm4vImageProcessor，Qwen2VL 系列架構）版本的 image placeholder token 數量
    一致性檢查，跟 assert_image_token_alignment()（Pixtral 專用）不只公式不同、連 processor
    輸出的欄位名稱都不同，兩者不可互換：

    - Pixtral：processor 輸出 image_sizes（原始像素高寬），每列 num_width_tokens 個 [IMG] 後面
      還多插一個 [IMG_BREAK]/[IMG_END]，公式是 num_height_tokens * (num_width_tokens + 1)。
    - GLM-4.1V：processor 輸出 image_grid_thw（每張圖的 (t, h, w) patch 網格數，已經是
      像素尺寸 // patch_size 之後的值，不是原始像素高寬），整張圖只用單一個
      <|begin_of_image|> ... <|end_of_image|> 包住、中間放 t*h*w // merge_size**2 個 image
      token，沒有逐列插入的 break token，公式對應
      transformers/models/glm4v/processing_glm4v.py 的
      `num_image_tokens = image_grid_thw[i].prod() // merge_size**2`。
    """
    if "image_grid_thw" not in batch:
        return  # 這個 batch 沒有圖片，略過

    merge_length = merge_size**2
    input_ids = batch["input_ids"]
    image_grid_thw = batch["image_grid_thw"]

    for i, example in enumerate(examples):
        t, h, w = (int(v) for v in image_grid_thw[i])
        expected_tokens = (t * h * w) // merge_length

        row = input_ids[i]
        actual_tokens = sum(int((row == tok_id).sum().item()) for tok_id in image_token_ids)

        if actual_tokens != expected_tokens:
            image_obj = example.get("image")
            image_path = getattr(image_obj, "filename", None)
            raise AssertionError(
                "image token 數量與圖片尺寸換算結果不一致（GLM-4.1V 版本）：\n"
                f"  batch 內索引={i}, image_path={image_path}, background={example.get('background')}\n"
                f"  image_grid_thw=({t},{h},{w}), merge_size={merge_size}\n"
                f"  預期 token 數={expected_tokens}, 實際 token 數={actual_tokens}"
            )
