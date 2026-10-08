"""GPU/CPU 資源偵測：決定模型載入時的 max_memory 上限，以及訓練時的 batch size 等設定。"""

import os

import psutil
import torch


def detect_max_memory(gpu_fraction: float = 0.8, cpu_fraction: float = 0.8) -> dict:
    """偵測目前實際可用的 GPU / CPU 記憶體，只取其中一部分（預留緩衝給 activation 等）。

    取代寫死的數字（例如固定 10GB/20GB）：寫死的值在不同機器、或同一機器上有其他行程
    占用 GPU 時都可能不合適 —— 太保守會讓模型被不必要地 CPU offload、拖慢訓練，太激進
    則會在載入或訓練時 OOM。
    """
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
        max_memory[i] = f"{free_bytes * gpu_fraction / 1024**3:.1f}GB"

    try:
        cpu_available_gb = psutil.virtual_memory().available / 1024**3
    except Exception as e:
        print(f"⚠️ 無法透過 psutil 偵測可用 CPU 記憶體，改用保守預設值：{e}")
        cpu_available_gb = 8.0
    max_memory["cpu"] = f"{cpu_available_gb * cpu_fraction:.1f}GB"

    print(f"[max-memory-detect] -> {max_memory}")
    return max_memory


def detect_resources_and_configure(target_effective_batch_size: int = 16):
    """依目前可用的 GPU / CPU 資源，動態決定 batch size 等訓練設定，避免 OOM。"""
    cpu_count = os.cpu_count() or 1
    try:
        cpu_ram_gb = psutil.virtual_memory().available / 1024**3
    except Exception as e:
        print(f"⚠️ 無法透過 psutil 偵測可用 CPU 記憶體，改用保守預設值：{e}")
        cpu_ram_gb = 8.0

    gpu_free_gb = []
    for i in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
        gpu_free_gb.append(free_bytes / 1024**3)
    # 用「可用記憶體最小的那張卡」當基準，確保多卡情景下每張卡都不會被打爆
    min_gpu_free_gb = min(gpu_free_gb) if gpu_free_gb else 0.0

    if min_gpu_free_gb >= 60:
        per_device_bs = 8
    elif min_gpu_free_gb >= 40:
        per_device_bs = 4
    elif min_gpu_free_gb >= 24:
        per_device_bs = 2
    else:
        per_device_bs = 1
    # 維持 effective batch size 大致不變，用 gradient accumulation 補回去
    grad_accum = max(1, target_effective_batch_size // per_device_bs)

    # worker 數量同時受 CPU 核心數與可用 RAM 限制（RAM 太小時，過多 worker 會同時 prefetch
    # 多份 batch 的圖片，反而加重記憶體壓力）
    num_workers = max(1, min(8, cpu_count // 2))
    if cpu_ram_gb < 16:
        num_workers = min(num_workers, 2)

    print(
        f"[resource-detect] CPU cores={cpu_count}, 可用 RAM={cpu_ram_gb:.1f}GB, "
        f"各 GPU 可用記憶體={[f'{g:.1f}GB' for g in gpu_free_gb]}"
    )
    print(
        f"[resource-detect] -> per_device_train_batch_size={per_device_bs}, "
        f"gradient_accumulation_steps={grad_accum}, dataloader_num_workers={num_workers}"
    )
    return per_device_bs, grad_accum, num_workers
