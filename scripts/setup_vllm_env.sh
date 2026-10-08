#!/bin/bash
# setup_vllm_env.sh — 從 flash-attn 訓練 env（預設 Mistral3-fa，見 setup_flash_attn_env.sh）clone 出
# 一個新 env（預設 Mistral3-fa-vllm），在新 env 裡加裝 vLLM，讓同一個 env 同時能訓練與推論。
# 來源 env 不會被修改，vLLM 有問題時可以直接退回來源 env 繼續訓練。
#
# 為什麼不是直接 `pip install vllm==0.28.0`（2026-10-07 調查結果）：
#   - PyPI 上的 vllm 預設 wheel 是 CUDA 13 build，會一併拉進 nvidia-cuda-runtime 13 等套件；這台機器
#     driver 535 只支援到 CUDA 12.2，import 得起來但上 GPU 就會失敗。
#   - vllm 0.27.0~0.31.0 都釘 torch==2.13.0，GitHub release 另外提供 `+cu129` 的 CUDA 12 build；
#     0.28.0+cu129 的編譯擴充用 stable libtorch ABI、連結 libcudart.so.12，可以直接跑在
#     torch 2.13.0+cu126 上（實測 vllm 自己的 kernel 在 GPU 上算出正確結果），不需要換 torch。
#   - 但裝 vllm 時會順帶裝 torchaudio 2.11.0 / torchcodec 0.17.0 的 PyPI 預設版（CUDA 13 build，
#     連結 libcudart.so.13 / libnvrtc.so.13）。transformers 載入時會 import torchaudio，於是
#     `vllm serve` 一啟動就 `OSError: libcudart.so.13: cannot open shared object file`（實測），
#     這裡裝完 vllm 後把這兩個換成同版本的 cu126 build（--no-deps，不動 torch）。
#   - 版本選 0.28.0：跟 requirements.txt / 既有 Mistral env 同版本，markdown/USAGE.md 的
#     vllm serve 參數是在這個版本上驗證過的。唯一會改到既有套件的是 protobuf 7.36.2 -> 6.33.6，
#     已驗證 wandb 0.30.0 離線記錄正常。
#
# 用法：
#   ./scripts/setup_vllm_env.sh
#
# 環境變數（全部可覆寫）：
#   SRC_ENV        來源 conda env 名稱，預設 Mistral3-fa（不會被修改）
#   DST_ENV        新 env 名稱，預設 Mistral3-fa-vllm（已存在則直接沿用，不重新 clone）
#   CONDA_ROOT     miniconda 安裝路徑，預設 /root/miniconda3
#   VLLM_VERSION   預設 0.28.0（需要是 GitHub release 有 +cu129 wheel、且釘 torch==2.13.0 的版本）
#   TORCH_CUDA_TAG 用來換掉 CUDA 13 build 的 PyTorch index 標籤，預設 cu126（需與 env 的 torch 一致）

set -euo pipefail

SRC_ENV="${SRC_ENV:-Mistral3-fa}"
DST_ENV="${DST_ENV:-Mistral3-fa-vllm}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu126}"

DST_PREFIX="$CONDA_ROOT/envs/$DST_ENV"
PY="$DST_PREFIX/bin/python"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

echo "=== 1/4 clone $SRC_ENV -> $DST_ENV"
if [ -x "$PY" ]; then
    echo "$DST_ENV 已存在，沿用（不重新 clone）"
else
    "$CONDA_ROOT/bin/conda" create -y -q --clone "$SRC_ENV" -n "$DST_ENV"
fi
SITE_PACKAGES=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
torch_before=$("$PY" -c "import torch; print(torch.__version__)")
echo "torch $torch_before"

echo "=== 2/4 安裝 vllm ${VLLM_VERSION}+cu129"
"$PY" -m pip install -q "$VLLM_WHEEL"

echo "=== 3/4 把 vllm 連帶裝進來的 CUDA 13 build 換成 $TORCH_CUDA_TAG build"
# 版本號沿用 vllm 剛裝好的版本，只換 CUDA build（+cu126），--no-deps 確保不會動到 torch。
torchaudio_ver=$("$PY" -c "import importlib.metadata as m; print(m.version('torchaudio').split('+')[0])")
torchcodec_ver=$("$PY" -c "import importlib.metadata as m; print(m.version('torchcodec').split('+')[0])")
"$PY" -m pip install -q --no-deps --force-reinstall --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_TAG" \
    "torchaudio==${torchaudio_ver}+${TORCH_CUDA_TAG}" "torchcodec==${torchcodec_ver}+${TORCH_CUDA_TAG}"

# 掃一遍整個 env，確認沒有任何 .so 還連結 CUDA 13 runtime（driver 535 載入不了）。
cuda13_libs=$(find "$SITE_PACKAGES" -name "*.so*" -type f 2>/dev/null | while read -r f; do
    readelf -d "$f" 2>/dev/null | grep -qE "lib(cudart|nvrtc|cublas|cublasLt|nvJitLink)\.so\.13" && echo "$f"
done || true)
if [ -n "$cuda13_libs" ]; then
    echo "以下函式庫仍連結 CUDA 13 runtime，這台機器的 driver 載入不了："
    echo "$cuda13_libs"
    exit 1
fi

echo "=== 4/4 驗證"
"$PY" - "$torch_before" <<'EOF'
import sys

import torch
import torchaudio  # noqa: F401  transformers 會 import，CUDA build 不對時這裡就會炸
import flash_attn
import transformers
import vllm
import vllm._C_stable_libtorch  # noqa: F401

assert torch.__version__ == sys.argv[1], f"torch 被換掉了：{sys.argv[1]} -> {torch.__version__}"
assert torch.cuda.is_available(), "torch 看不到 GPU"
x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(x)
torch.ops._C.rms_norm(out, x, w, 1e-6)
ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
diff = (out.float() - ref).abs().max().item()
assert diff < 2e-2, f"vllm rms_norm kernel 結果不對：max diff {diff}"
print(f"torch {torch.__version__} / flash_attn {flash_attn.__version__} / transformers {transformers.__version__} "
      f"/ vllm {vllm.__version__}，vllm GPU kernel max diff {diff:.4f}")
EOF
echo "完成：$DST_PREFIX/bin/vllm（啟動參數見 markdown/USAGE.md 的 3/4 vllm serve）"
