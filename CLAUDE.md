# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This project fine-tunes a Mistral3-family VLM (Ministral-3-8B-Instruct-2512 / Mistral-Small-3.2-24B, `Mistral3ForConditionalGeneration` + `PixtralProcessor`) with LoRA/SFT (TRL). The training data is synthetic Taiwanese traffic-collision diagrams (肇事碰撞構圖). The model learns to identify collision type, direction, and casualties in Traditional Chinese. Merged models are served with vLLM (an OpenAI-compatible API) and shown through a FastAPI demo.

Code comments, docstrings, logs, and docs are written in Traditional Chinese. Match that when editing. Comments are long and explain *why*, often citing incidents that were reproduced on this machine. Keep them when refactoring.

The repo is not a git repo, and the root `README.md` is an unedited GitLab template. The real usage docs are in `markdown/USAGE.md`.

## Environment

- Use the conda env at `/root/miniconda3/envs/fine_tune_vlm/bin/python` (Python 3.10). The vLLM CLI is in that env's `bin/vllm`. The pinned stack (transformers 5.x, trl, peft, vllm) is in `requirements.txt`. `requirements-fine_tune_vlm.txt` is a raw `pip freeze` snapshot kept for reference and is not meant to be installed.
- Run all scripts from the repo root. Scripts in `scripts/` add the repo root to `sys.path` themselves, so `utils` imports resolve.
- Base models live under `models/<model_id>`. Training runs write to `Output/<model_id>-lora-run<N>/`.
- `configs/secrets.yaml` holds `wandb_api_key` and must never be committed or hard-coded. Copy it from `secrets.yaml.example`. For local runs, set `WANDB_MODE=offline` and put any string as the key.
- No test suite or linter is configured. Verification is the smoke test below.

## Common commands

```bash
PY=/root/miniconda3/envs/fine_tune_vlm/bin/python

# Generate the synthetic dataset (pick a generator version; see --help)
$PY scripts/make_data.py --help

# End-to-end smoke test: train 10 steps → merge → vllm serve → test inference (cleans up Output/ afterwards)
./scripts/smoke_test_train_merge_vllm.sh
SMOKE_TEST_KEEP_OUTPUT=1 ./scripts/smoke_test_train_merge_vllm.sh   # keep the run dir

# Formal training under the auto-restart supervisor loop (log: formal_finetune_ministral.log)
./scripts/run_formal_finetune.sh --model_id Ministral-3-8B-Instruct-2512 --num_train_epochs 10 \
  --eval_strategy steps --eval_steps 100 --data_root data/collision_diagram/synthetic_tune_v2 --new_run

# Monitor a running training job (process, GPU, disk, live tqdm progress via /proc/<pid>/fd)
./scripts/monitor_progress.sh -w

# Merge any LoRA checkpoint by hand. Always pass --model_dir explicitly; its default may be a different model family.
$PY scripts/merge_lora_checkpoint.py --checkpoint "$RUN_DIR" --merged_path "$RUN_DIR/merged" \
  --model_dir models/Ministral-3-8B-Instruct-2512
```

`markdown/USAGE.md` has the step-by-step manual version of the smoke test (including the exact `vllm serve` flags: `--tokenizer_mode mistral --config_format mistral --tool-call-parser mistral`, `VLLM_USE_FLASHINFER_SAMPLER=0`) and the demo-server curl tests.

## Architecture

**Training scripts** (`scripts/fine_tuning_vlm_trl_traffic_tune_ministral_*.py`): each is a long, top-level, numbered-section script (`# 0. 命令列參數` … `# 9. 合併`).
- `_fastmerge.py` and `_simplified.py` are deliberate independent copies that behave identically. `_simplified.py` only refactors section 9 into named functions. Maintain them separately: do not deduplicate them or "sync" one into the other unless asked.
- CLI args and the `TrainingConfig` defaults are in `utils/fine_tuning/config.py`. The default `--model_id` is not Ministral, so pass `--model_id` explicitly.
- `--resource_mode auto|manual` decides whether batch size, grad accumulation, and workers come from `utils/fine_tuning/resources.py` auto-detection or from CLI flags.
- `--skip_merge` saves only the adapter.

**Run management** (`utils/fine_tuning/run_management.py`): output directories are named by run index, not by date.
- Without `--new_run`, a run resumes the highest-numbered run from its last *complete* checkpoint. Half-written checkpoints are skipped.
- `--new_run` starts `run{N+1}` from the base model.
- This is what lets `run_formal_finetune.sh` restart after crashes. That script exists because the GPU driver has caused random SIGSEGV/SIGKILL and hangs. It also kills the job after a stall timeout and stops after `MAX_CONSECUTIVE_FAILURES`.

**Known pitfalls encoded in the code (do not "simplify" away):**
- The Ministral checkpoint on disk is FP8. Load it with `FineGrainedFP8Config(dequantize=True)` so it becomes bf16, and never stack bitsandbytes on top. At merge time the config must match the base model's `weight_block_size` (None for Ministral). If it doesn't, transformers silently loads raw FP8 weights and the merged model outputs garbage in vLLM.
- Merge path: `save_pretrained()` re-quantizes to FP8 through `revert_weight_conversion()`. The fastmerge variant calls `revert_weight_conversion()` itself, fixes the state_dict in memory, and writes once. When copying tokenizer files to `merged/`, never copy the base model's `config.json`. It still carries the FP8 `quantization_config` and would overwrite the clean bf16 config.
- `torch.multiprocessing` sharing strategy is `file_system` because `/dev/shm` is small in the container. The script asserts the start method is `fork`, because `spawn` would make every DataLoader worker reload the model, and this machine has no swap.
- Attention is `sdpa` (no nvcc available, so no flash-attn). transformers 5.x renamed `torch_dtype` to `dtype` and dropped `warmup_ratio`, which is now converted to `warmup_steps`.
- `device_map="auto"` can silently offload layers to the CPU. The script checks for this.

**Data generation** (`utils/ds_utils/`, entry point `scripts/make_data.py`):
- `CollisionDiagramBase` (`base.py`) handles base maps, collision types, and coordinates. `rendering.py` composites icons and `domain/` holds the traffic rules and geometry.
- `generators/` has one subclass per dataset format version: `CollisionDescriptionV1/V2/V3`, `CollisionDiagramMix`, and `CollisionDiagramTuneV1/V2`. They are registered in `GENERATOR_REGISTRY` in `make_data.py`, and `generators/generators.md` documents them.
- `facade.CollisionDiagram` recombines them for backward compatibility.
- Output goes to `data/collision_diagram/<dataset>/{train,test}/*.jsonl` with `image`/`text` records. `make_simple_data.py` is a sibling variant.

**Other utils:**
- `utils/accuracy_utils.py` scores model output against ground truth.
- `utils/report_utils.py` and `utils/gen_utils.py` cover reporting and generation helpers.
- `utils/vl_utils.py` holds the VLM input helpers.

**Demo** (`Demo/`; the docs call it `demo/`):
- `inference_server.py` is a FastAPI app with the routes `GET /`, `POST /analyze` → 303 to `/result/{id}`, `/vlm_response`, and `/multi_response`.
- The current version does not call the VLM live. It matches uploaded images by pHash (`image_lookup.py`) against precomputed results in `Demo/test.jsonl`, with a simulated delay.
- Start it with `python Demo/inference_server.py --port 8090` from the repo root. `demo_ori/` is the earlier version, which needs vLLM running on `:8000`.

`Progress/` holds progress reports and `openspec/` is an unconfigured OpenSpec setup. `notebooks/` contains inference experiments.
