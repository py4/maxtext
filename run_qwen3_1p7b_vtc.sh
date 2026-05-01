#!/bin/bash
# VTC-faithful Qwen3-1.7B GSM8K GRPO recipe in MaxText.
# All GPU-matching params are env-overridable. Defaults match the GPU
# recipe at /home/pooyam_google_com/nemo_rl/85_repro/configs/repro_grpo_qwen3_1p7b_gsm8k.yaml
# unless noted.
#
# Run from inside the controller pod, after kubectl-cp + pip install -e maxtext.
#   bash /workspace/maxtext/run_qwen3_1p7b_vtc.sh
#
# Common overrides:
#   NUM_BATCHES=150            # train steps (GPU best result @ 150)
#   EVAL_INTERVAL=50           # eval every N train steps (uses VTC reward)
#   EVAL_BATCH_SIZE=32         # eval throughput (re-batched in our eval patch)
#   BATCH_SIZE=4               # training prompts per step (matches GPU)
#   ROLLOUT_TEMP=1.0           # GPU uses 1.0, vLLM default is 0.6
#   RUN_NAME=qwen3-1p7b-grpo-vtc-runX
#
# Mechanism toggle (off by default — see scratchpad_maxtext.md "v11" section):
#   ENABLE_ROLLOUT_OLDLOGPS=1  # feed vLLM rollout logprobs as old_per_token_logps
#                                so PPO ratio ≠ 1 (real clipping). Costs ~+50%
#                                step time; bought only +0.79pp post-RL VTC in v11
#                                vs v10. Default off.

set -ex

export JAX_PLATFORMS="${JAX_PLATFORMS:-proxy,cpu}"

# ---- Identity / output paths ----
RUN_NAME="${RUN_NAME:-qwen3-1.7b-grpo-vtc-run1}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-gs://vtc-pathways-scratch-infinipod-shared-dev/runs/qwen3-1.7b-grpo-vtc}"
TENSORBOARD_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}/tensorboard"
echo "TENSORBOARD: $TENSORBOARD_DIR"

# ---- Training shape (GPU parity) ----
BATCH_SIZE="${BATCH_SIZE:-4}"               # = num_prompts_per_step (GPU: 4)
NUM_BATCHES="${NUM_BATCHES:-150}"           # = max_num_steps (GPU: 200; we stop at 150)
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"     # GPU: 8
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"       # GPU effective: 2 (we use 1 due to tunix vLLM kw bug)
RNG_SEED="${RNG_SEED:-42}"                  # GPU: 42

# ---- Sequence lengths ----
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"   # GPU max_total_sequence_length: 1024
MAX_PREFILL="${MAX_PREFILL:-256}"                # full GSM8K test prompts fit
MAX_GEN=$((MAX_TARGET_LENGTH - MAX_PREFILL))     # 768 generation tokens (GPU: 768)

# ---- Optimizer (GPU parity) ----
LR="${LR:-2e-7}"
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-500}"
WARMUP_FRAC="${WARMUP_FRAC:-0.25}"
ADAM_B1="${ADAM_B1:-0.9}"           # GPU: 0.9
ADAM_B2="${ADAM_B2:-0.999}"         # GPU: 0.999
ADAM_WD="${ADAM_WD:-0.01}"          # GPU: 0.01
GRAD_CLIP="${GRAD_CLIP:-1.0}"       # GPU: 1.0

# ---- GRPO algo (GPU parity) ----
GRPO_BETA="${GRPO_BETA:-0.04}"      # GPU reference_policy_kl_penalty
GRPO_EPSILON="${GRPO_EPSILON:-0.2}" # GPU ratio_clip
# KL estimator (mse_kl == GPU's k2) and ddof=0 advantages applied via
# my_train_rl.py monkey-patches.

# ---- Rollout sampling (GPU parity) ----
ROLLOUT_TEMP="${ROLLOUT_TEMP:-1.0}"        # GPU: 1.0; vLLM default is 0.6 from generation_config
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"      # GPU: 1.0
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"       # GPU: null (-1 disables top_k in MaxText)

# ---- Eval ----
# Eval count is computed so num_test_batches × BATCH_SIZE ≥ 1319 (full GSM8K test).
# Inside my evaluate_rl patch, the test dataset is re-batched to EVAL_BATCH_SIZE
# for vLLM throughput.
NUM_TEST_BATCHES="${NUM_TEST_BATCHES:-330}"        # 330 × 4 = 1320 ≥ 1319
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"           # vLLM rebatch size in patched eval
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"               # eval every N train steps (= GPU val_period)
EVAL_STEPS="${EVAL_STEPS:-20}"                     # tunix intermediate-eval batches per cycle (each = bs prompts × num_gen). Default 20 → ~80 prompts/eval ~3 min

# ---- vLLM (TPU specific) ----
HBM_UTIL="${HBM_UTIL:-0.50}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"      # tensor parallel for rollout
ROLLOUT_DP="${ROLLOUT_DP:--1}"     # auto data parallel

# Export for my_train_rl.py to read
export EVAL_BATCH_SIZE
export EVAL_INTERVAL

python3 -u /workspace/maxtext/my_train_rl.py \
  /workspace/maxtext/src/maxtext/configs/post_train/rl.yml \
  model_name=qwen3-1.7b \
  tokenizer_path=Qwen/Qwen3-1.7B \
  load_parameters_path=gs://vtc-pathways-scratch-infinipod-shared-dev/models/qwen3-1.7b/0/items \
  dataset_name=gsm8k \
  run_name="$RUN_NAME" \
  base_output_directory="$BASE_OUTPUT_DIR" \
  hf_access_token="$HF_TOKEN" \
  chat_template_path=/tmp/gsm8k_qwen3.json \
  init_weights_seed="$RNG_SEED" \
  data_shuffle_seed="$RNG_SEED" \
  batch_size="$BATCH_SIZE" \
  num_batches="$NUM_BATCHES" \
  num_test_batches="$NUM_TEST_BATCHES" \
  eval_interval="$EVAL_INTERVAL" \
  eval_steps="$EVAL_STEPS" \
  learning_rate="$LR" \
  learning_rate_schedule_steps="$LR_SCHEDULE_STEPS" \
  warmup_steps_fraction="$WARMUP_FRAC" \
  adam_b1="$ADAM_B1" \
  adam_b2="$ADAM_B2" \
  adam_weight_decay="$ADAM_WD" \
  gradient_clipping_threshold="$GRAD_CLIP" \
  max_target_length="$MAX_TARGET_LENGTH" \
  max_prefill_predict_length="$MAX_PREFILL" \
  decode_sampling_temperature="$ROLLOUT_TEMP" \
  decode_sampling_nucleus_p="$ROLLOUT_TOP_P" \
  decode_sampling_top_k="$ROLLOUT_TOP_K" \
  rl.grpo_beta="$GRPO_BETA" \
  rl.grpo_epsilon="$GRPO_EPSILON" \
  rl.num_generations="$NUM_GENERATIONS" \
  rl.num_iterations="$NUM_ITERATIONS" \
  hbm_utilization_vllm="$HBM_UTIL" \
  checkpoint_storage_concurrent_gb=48 \
  load_checkpoint_only_once=True \
  jax_cache_dir=gs://vtc-pathways-scratch-infinipod-shared-dev/jax_cache/qwen3-1.7b-grpo-vtc \
  rollout_tensor_parallelism="$ROLLOUT_TP" \
  rollout_data_parallelism="$ROLLOUT_DP" \
  "vllm_hf_overrides={architectures: [MaxTextForCausalLM]}" \
  "vllm_additional_config={maxtext_config: {model_name: qwen3-1.7b, allow_split_physical_axes: true, log_config: false, weight_dtype: bfloat16}}"
