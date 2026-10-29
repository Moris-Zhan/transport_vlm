# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This project fine-tunes the Mistral-Small-3.2-24B-Instruct-2506 VLM (`Mistral3ForConditionalGeneration` + `PixtralProcessor`, bf16 weights) with LoRA/SFT (TRL). The training data is synthetic Taiwanese traffic-collision diagrams (肇事碰撞構圖). The model learns to identify collision type, direction, and casualties in Traditional Chinese. Merged models are served with vLLM (an OpenAI-compatible API) and shown through a FastAPI demo.

Code comments, docstrings, logs, and docs are written in Traditional Chinese. Match that when editing. Comments are long and explain *why*, often citing incidents that were reproduced on this machine. Keep them when refactoring.

`README.md` is the project overview. Step-by-step usage is in `markdown/USAGE.md`, and the H100 environment/training handoff notes are in `markdown/HANDOFF-2026-10-07-H100訓練.md`.

## Environment

- Use the conda env at `/root/miniconda3/envs/Mistral3-fa-vllm/bin/python` (Python 3.10, torch 2.13.0+cu126, flash-attn 2.8.3.post1, vllm 0.28.0+cu129). It is used for both training and inference. The vLLM CLI is in that env's `bin/vllm`. It was built in layers (`Mistral3` → `Mistral3-fa` → `Mistral3-fa-vllm`) by `scripts/setup_flash_attn_env.sh` and `scripts/setup_vllm_env.sh`; the machine's driver 535 only supports CUDA 12.2, so CUDA 13 builds of torch/vllm cannot see the GPU.
- The curated package list is `requirements.txt`. `requirements-fine_tune_vlm.txt` and `requirements-Mistral3-fa-vllm.txt` are raw `pip freeze` snapshots kept for reference and are not meant to be installed.
- Run all scripts from the repo root. Scripts in `scripts/` add the repo root to `sys.path` themselves, so `utils` imports resolve.
- Base models live under `models/<model_id>`. Training runs write to `Output/<model_id>-lora-run<N>/`.
- Mistral-Small needs both GPUs: set `CUDA_VISIBLE_DEVICES=0,1` from outside. Check `nvidia-smi` before using GPU0; an external process invisible to this container has occupied it before.
- `configs/secrets.yaml` holds `wandb_api_key`, is in `.gitignore`, and must never be committed or hard-coded. Copy it from `secrets.yaml.example`. For local runs, set `WANDB_MODE=offline` and put any string as the key.
- No test suite or linter is configured. Verification is the manual train → merge → vLLM → inference flow in `markdown/USAGE.md`.

## Common commands

```bash
PY=/root/miniconda3/envs/Mistral3-fa-vllm/bin/python

# Generate the synthetic dataset (pick a generator version; see --help)
$PY scripts/make_data.py --help

# Short training run (adapter only)
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
  $PY scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py --max_steps 10 --new_run

# Formal training under the auto-restart supervisor loop
CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_formal_finetune.sh --model_id Mistral-Small-3.2-24B-Instruct-2506 --new_run

# Monitor a running training job (process, GPU, disk, live tqdm progress via /proc/<pid>/fd)
./scripts/monitor_progress.sh -w

# Merge a LoRA run dir or checkpoint-N (CPU only, ~7 min)
CUDA_VISIBLE_DEVICES="" $PY scripts/merge_lora_mistral_small.py --checkpoint "$RUN_DIR" --merged_path "$RUN_DIR/merged"

# Serve the merged model
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 vllm serve "$RUN_DIR/merged" --tokenizer "$RUN_DIR/merged" \
  --load_format safetensors --enable-auto-tool-choice --tool-call-parser mistral --max-model-len 8192 --port 8000
```

`markdown/USAGE.md` has the full manual flow (including the inference test script) and the demo-server curl tests.

## Architecture

**Training script** (`scripts/fine_tuning_vlm_trl_traffic_tune_mistral_small.py`): a long, top-level, numbered-section script. It only accepts `--model_id Mistral-Small-3.2-24B-Instruct-2506` and saves only the LoRA adapter + processor (`trainer.save_model()`); merging is done separately by `scripts/merge_lora_mistral_small.py`. `scripts/fine_tuning_vlm_trl_traffic_tune_v2.py` is the earlier single-file script it was simplified from (date-based output dirs, no run management).
- CLI args and the `TrainingConfig` defaults are in `utils/fine_tuning/config.py`. The default `--model_id` is Mistral-Small. Some help text and choices there still mention the removed Ministral scripts; leave them unless asked.
- `--resource_mode auto|manual` decides whether batch size, grad accumulation, and workers come from `utils/fine_tuning/resources.py` auto-detection or from CLI flags.
- `--attn_implementation` defaults to `flash_attention_2`; `sdpa` is the fallback for envs without flash-attn (measured ~1.9× slower, same loss curve).

**Run management** (`utils/fine_tuning/run_management.py`): output directories are named by run index, not by date.
- Without `--new_run`, a run resumes the highest-numbered run from its last *complete* checkpoint. Half-written checkpoints are skipped.
- `--new_run` starts `run{N+1}` from the base model.
- This is what lets `run_formal_finetune.sh` restart after crashes. That script exists because the GPU driver has caused random SIGSEGV/SIGKILL and hangs. It also kills the job after a stall timeout and stops after `MAX_CONSECUTIVE_FAILURES`.

**Known pitfalls encoded in the code (do not "simplify" away):**
- `torch.multiprocessing` sharing strategy is `file_system` because `/dev/shm` is small in the container. The start method must be `fork`, because `spawn` would make every DataLoader worker reload the model, and this machine has no swap.
- transformers 5.x renamed `torch_dtype` to `dtype` and dropped `warmup_ratio`, which is now converted to `warmup_steps`.
- `device_map="auto"` can silently offload layers to the CPU. The script checks for this.

**Data generation** (`utils/ds_utils/`, entry point `scripts/make_data.py`):
- `CollisionDiagramBase` (`base.py`) handles base maps, collision types, and coordinates. `rendering.py` composites icons and `domain/` holds the traffic rules and geometry.
- `generators/` has one subclass per dataset format version: `CollisionDescriptionV1/V2/V3`, `CollisionDiagramMix`, and `CollisionDiagramTuneV1/V2`. They are registered in `GENERATOR_REGISTRY` in `make_data.py`, and `generators/generators.md` documents them.
- `facade.CollisionDiagram` recombines them for backward compatibility.
- Output goes to `data/collision_diagram/synthetic_<dataset_tag>/{train,test}/*.jsonl` with `image`/`text` records. `make_simple_data.py` is a sibling variant.

**Other utils:**
- `utils/accuracy_utils.py` scores model output against ground truth.
- `utils/report_utils.py` and `utils/gen_utils.py` cover reporting and generation helpers.
- `utils/vl_utils.py` holds the VLM input helpers.

**Demo** (`Demo/`):
- `inference_server.py` is a FastAPI app with the routes `GET /`, `POST /analyze` → 303 to `/result/{id}`, `/vlm_response`, and `/multi_response`.
- It calls the vLLM server at `http://localhost:8000/v1` live and calls `client.models.list()` at import time, so vLLM must be running before it starts.
- Start it with `python Demo/inference_server.py --port 8090` from the repo root. `test_API.py` calls the remote production deployment, not the local server.
- `start.sh` is the container entrypoint: it starts `vllm serve` on a merged model, waits for `/health`, then starts the frontend with uvicorn (`--root-path` for the ingress prefix). Its paths are hard-coded for the deployment machine.

`notebooks/` contains inference experiments.
