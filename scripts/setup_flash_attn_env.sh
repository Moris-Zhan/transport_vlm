#!/bin/bash
# setup_flash_attn_env.sh — 從既有的訓練 conda env（預設 Mistral3）clone 出一個新 env（預設
# Mistral3-fa），在新 env 裡從原始碼編譯 flash-attn，不動到來源 env 的任何套件版本。
#
# 為什麼要自己編譯（2026-10-07 調查結果）：
#   - 訓練 env 用 torch 2.13.0+cu126（這台機器 driver 535 只支援到 CUDA 12.2，cu130 版 torch
#     看不到 GPU，cu12x 版靠 CUDA minor version compatibility 可以跑）。
#   - flash-attn 官方 GitHub release 的預編譯 wheel 最新只到 torch 2.8；HF kernel hub
#     (kernels-community/flash-attn2) 也只到 torch 2.9。flash_attn 是跟 torch C++ ABI 綁死的
#     C++/CUDA extension，torch2.8 的 wheel 裝在 torch 2.13 上 import 會直接
#     `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`（實測）。
#   - conda 的 cuda-nvcc 套件會強制連帶安裝 conda 版 gcc，且跟 env 裡既有的 ld_impl 衝突解不開，
#     所以改用 NVIDIA 官方 redistributable tarball，解壓到新 env 底下的 cuda-12.6/ 當 CUDA_HOME，
#     不經過套件管理器、不改動 env 既有套件，搭配系統 gcc 13.3 編譯。
#
# 用法：
#   ./scripts/setup_flash_attn_env.sh
#
# 環境變數（全部可覆寫）：
#   SRC_ENV            來源 conda env 名稱，預設 Mistral3（不會被修改）
#   DST_ENV            新 env 名稱，預設 Mistral3-fa（已存在則直接沿用，不重新 clone）
#   CONDA_ROOT         miniconda 安裝路徑，預設 /root/miniconda3
#   CUDA_REDIST_VER    NVIDIA redist 清單版本，預設 12.6.3（需與 torch 的 cu126 對應）
#   FLASH_ATTN_VERSION 預設 2.8.3.post1
#   FLASH_ATTN_CUDA_ARCHS  只編譯哪些 GPU 架構，預設 90（H100）；全部架構編譯時間會多好幾倍
#   MAX_JOBS / NVCC_THREADS  平行編譯數，預設 32 / 4（108 核、2TB RAM 的機器實測約 30 分鐘）
#
# 編完後的使用方式：
#   FINETUNE_PYTHON=$CONDA_ROOT/envs/$DST_ENV/bin/python ...（或直接用該 python 執行訓練腳本）
#   訓練腳本 fine_tuning_vlm_trl_traffic_tune_ministral_simplified.py 預設
#   --attn_implementation flash_attention_2；在沒有 flash_attn 的 env 執行請改傳 sdpa。

set -euo pipefail

SRC_ENV="${SRC_ENV:-Mistral3}"
DST_ENV="${DST_ENV:-Mistral3-fa}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CUDA_REDIST_VER="${CUDA_REDIST_VER:-12.6.3}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
export MAX_JOBS="${MAX_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

DST_PREFIX="$CONDA_ROOT/envs/$DST_ENV"
PY="$DST_PREFIX/bin/python"
CUDA_DIR="$DST_PREFIX/cuda-${CUDA_REDIST_VER%.*}"
REDIST_BASE="https://developer.download.nvidia.com/compute/cuda/redist"
# 編譯 flash-attn 需要的元件：nvcc 本身、cudart/cccl（thrust/cub）標頭，以及 torch 的
# ATen/cuda/CUDAContext.h 會 include 的 cublas/cusparse/cusolver/curand/nvrtc/nvtx 標頭。
CUDA_COMPONENTS=(cuda_nvcc cuda_cudart cuda_cccl libcublas libcusparse libcusolver libcurand cuda_nvrtc cuda_profiler_api cuda_nvtx)

echo "=== 1/5 clone $SRC_ENV -> $DST_ENV"
if [ -x "$PY" ]; then
    echo "$DST_ENV 已存在，沿用（不重新 clone）"
else
    "$CONDA_ROOT/bin/conda" create -y -q --clone "$SRC_ENV" -n "$DST_ENV"
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'torch 看不到 GPU'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI)"

echo "=== 2/5 下載 CUDA $CUDA_REDIST_VER redistributable 到 $CUDA_DIR"
if [ -x "$CUDA_DIR/bin/nvcc" ]; then
    echo "nvcc 已存在，略過下載"
else
    mkdir -p "$CUDA_DIR"
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT
    curl -sSfL -o "$tmp_dir/redist.json" "$REDIST_BASE/redistrib_${CUDA_REDIST_VER}.json"
    for comp in "${CUDA_COMPONENTS[@]}"; do
        rel=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['linux-x86_64']['relative_path'])" "$tmp_dir/redist.json" "$comp")
        curl -sSfL -o "$tmp_dir/$comp.tar.xz" "$REDIST_BASE/$rel"
        tar -xJf "$tmp_dir/$comp.tar.xz" -C "$CUDA_DIR" --strip-components=1
        echo "  ok $comp"
    done
fi
"$CUDA_DIR/bin/nvcc" --version | tail -1

echo "=== 3/5 安裝編譯工具（ninja/packaging/wheel）"
"$PY" -m pip install -q ninja packaging wheel

echo "=== 4/5 編譯 flash-attn $FLASH_ATTN_VERSION（ARCHS=$FLASH_ATTN_CUDA_ARCHS, MAX_JOBS=$MAX_JOBS）"
if "$PY" -c "import flash_attn, sys; sys.exit(0 if flash_attn.__version__ == sys.argv[1] else 1)" "$FLASH_ATTN_VERSION" 2>/dev/null; then
    echo "flash_attn $FLASH_ATTN_VERSION 已安裝且可 import，略過編譯"
else
    # FLASH_ATTENTION_FORCE_BUILD：不要讓 setup.py 去抓官方預編譯 wheel（版本對不上 torch 2.13）。
    # --no-build-isolation：一定要用 env 裡現有的 torch 編譯，ABI 才會對上。
    # --no-deps：不讓 pip 順手升級/降級任何既有套件。
    CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$DST_PREFIX/bin:$PATH" FLASH_ATTENTION_FORCE_BUILD=TRUE \
        "$PY" -m pip install -v --no-build-isolation --no-deps "flash-attn==$FLASH_ATTN_VERSION"
fi

echo "=== 5/5 驗證：import + 跟 SDPA 數值比對（bf16）"
"$PY" - <<'EOF'
import torch
import flash_attn
from flash_attn import flash_attn_func

print("flash_attn", flash_attn.__version__)
torch.manual_seed(0)
for causal in (False, True):
    q, k, v = [torch.randn(2, 1024, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    out = flash_attn_func(q, k, v, causal=causal)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v)), is_causal=causal
    ).transpose(1, 2)
    diff = (out.float() - ref).abs().max().item()
    out.sum().backward()
    assert diff < 2e-2, f"causal={causal} 與 SDPA 差異過大：{diff}"
    assert torch.isfinite(q.grad).all(), f"causal={causal} backward 出現非有限值"
    print(f"causal={causal} max|diff|={diff:.4f} backward OK")
EOF
echo "完成：$PY"
