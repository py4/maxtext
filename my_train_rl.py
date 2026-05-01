#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0

"""VTC-faithful Qwen3-1.7B GRPO recipe wrapper around MaxText's train_rl.

What this script does:
  1. Monkey-patches MaxText's three default reward functions
     (match_format_exactly / match_format_approximately / check_numbers)
     to a single VTC partial-credit reward (0 / 0.1 / 0.5 / 1.0).
  2. Monkey-patches tunix's grpo_loss_fn to:
       - compute the loss in float32 (avoid bf16 NaN that bit us in
         the pure-tunix attempt)
       - clamp per-token KL to [0, 50] to prevent overflow
  3. Defaults tunix's KL estimator to "mse_kl" (lower variance than the
     default "low_var_kl") and the loss aggregation mode to "token-mean"
     (per the user's working "Run 7" recipe).
  4. Forwards the original argv to MaxText's `train_rl.main`.

Usage (inside the controller pod, after `pip install -e /workspace/maxtext`):

  python3 -u /workspace/my_train_rl.py \
    /workspace/maxtext/src/maxtext/configs/post_train/rl.yml \
    model_name=qwen3-1.7b \
    tokenizer_path=Qwen/Qwen3-1.7B \
    load_parameters_path=gs://.../models/qwen3-1.7b/0/items \
    dataset_name=gsm8k \
    run_name=qwen3-1.7b-grpo-vtc-run1 \
    base_output_directory=gs://.../runs/qwen3-1.7b-grpo-vtc \
    hf_access_token=$HF_TOKEN \
    chat_template_path=/tmp/gsm8k_qwen3.json \
    batch_size=4 num_batches=200 num_test_batches=32 \
    learning_rate=2e-7 learning_rate_schedule_steps=500 warmup_steps_fraction=0.25 \
    max_prefill_predict_length=128 gradient_clipping_threshold=1.0 \
    adam_weight_decay=0.01 adam_b2=0.999 \
    rl.grpo_beta=0.04 rl.grpo_epsilon=0.2 \
    rl.num_generations=8 rl.num_iterations=2 \
    hbm_utilization_vllm=0.50 \
    checkpoint_storage_concurrent_gb=48 load_checkpoint_only_once=True \
    jax_cache_dir=gs://.../jax_cache/qwen3-1.7b-grpo-vtc \
    rollout_tensor_parallelism=4 rollout_data_parallelism=-1 \
    'vllm_hf_overrides={architectures: [MaxTextForCausalLM]}' \
    'vllm_additional_config={maxtext_config: {model_name: qwen3-1.7b, allow_split_physical_axes: true, log_config: false, weight_dtype: bfloat16}}'
"""

from __future__ import annotations

import re
import sys
from typing import Any, List

# ---------------------------------------------------------------------
# 1. VTC partial-credit reward (0 / 0.1 / 0.5 / 1.0)
# ---------------------------------------------------------------------


def _extract_boxed(text: str) -> str:
  ans = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = ans[-1] if ans else text
  boxed: List[str] = []
  stack: List[int] = []
  for i, ch in enumerate(content):
    if ch == "{":
      stack.append(i)
    elif ch == "}":
      if not stack:
        continue
      op = stack.pop()
      if content[:op].endswith(r"\boxed"):
        boxed.append(content[op + 1 : i])
  if boxed:
    return boxed[-1]
  m = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,]+)\s*\}?", content)
  if m:
    return m.group(1)
  raise ValueError("No boxed strings found")


def _vtc_score(response: str, expected: str) -> float:
  try:
    has_r = response.count("</reasoning>") == 1
    has_a = (
        response.count("<answer>") == 1
        and response.count("</answer>") == 1
    )
    ir = response.find("</reasoning>")
    iao = response.find("<answer>")
    iac = response.find("</answer>")
    ordered = has_r and has_a and (ir < iao < iac)
    fmt_ok = has_r and has_a and ordered
    given = _extract_boxed(response)
    cg = str(given).replace(",", "").strip()
    ce = str(expected).replace(",", "").strip()
    ok = cg == ce
    if not fmt_ok:
      return 0.5 if ok else 0.0
    return 1.0 if ok else 0.1
  except Exception:
    return 0.0


_VTC_PRINT_BUDGET = [3]


def vtc_partial_credit_reward(
    prompts: List[str],
    completions: List[str],
    tmvp_config: Any = None,
    answer: List[str] | None = None,
    **kwargs,
) -> List[float]:
  """Single replacement reward fn for MaxText's three default rewards.

  Honors MaxText's signature: (prompts, completions, tmvp_config, **kwargs).
  Pulls ground truth from the `answer` kwarg (passed by GRPOLearner).
  """
  if answer is None:
    raise ValueError("vtc_partial_credit_reward needs `answer` kwarg.")
  scores = []
  for completion, gold in zip(completions, answer):
    s = _vtc_score(completion, str(gold))
    scores.append(float(s))
    if _VTC_PRINT_BUDGET[0] > 0:
      _VTC_PRINT_BUDGET[0] -= 1
      try:
        print(
            "\n[vtc_partial_credit_reward] sample\n"
            "=== MODEL RESPONSE (truncated 1500 chars) ===\n"
            f"{completion[:1500]}\n"
            "=== END RESPONSE ===\n"
            f"ground_truth = {gold}\n"
            f"vtc_score    = {s}\n",
            flush=True,
        )
      except Exception:
        pass
  return scores


def _zero_reward(prompts, completions, tmvp_config=None, **kwargs):
  """No-op reward; MaxText sums all reward fns so this leaves the VTC one alone."""
  return [0.0 for _ in completions]


# ---------------------------------------------------------------------
# 2. MaxText reward override
# ---------------------------------------------------------------------


def _patch_maxtext_rewards():
  from maxtext.trainers.post_train.rl import utils_rl

  utils_rl.match_format_exactly = vtc_partial_credit_reward
  utils_rl.match_format_approximately = _zero_reward
  utils_rl.check_answer = _zero_reward
  utils_rl.check_numbers = _zero_reward
  print(
      "[my_train_rl] patched utils_rl reward fns:"
      " match_format_exactly -> vtc_partial_credit_reward;"
      " match_format_approximately/check_answer/check_numbers -> _zero_reward",
      flush=True,
  )


# ---------------------------------------------------------------------
# 2b. Raw VTC prompt (Fix E from prior Run 7 — bypass apply_chat_template
#     so Qwen3 doesn't trigger its native <think>/<answer> behavior and
#     instead follows our explicit <reasoning>...</reasoning> instructions).
# ---------------------------------------------------------------------


_VTC_PROMPT = (
    "Solve the following math problem.\n"
    "First, put your detailed step-by-step reasoning process inside"
    " <reasoning>...</reasoning> tags.\n"
    "Then, put your final numerical answer inside <answer>\\boxed{{}}</answer>"
    " tags. Do not put anything else in the answer tags.\n"
    "\n"
    "Problem: {question}\n"
    "<reasoning>\n"
)


def _patch_process_data():
  from maxtext.trainers.post_train.rl import utils_rl

  _orig_process_data = utils_rl.process_data

  def _patched_process_data(dataset_name, model_tokenizer, template_config, tmvp_config, x):
    out = _orig_process_data(dataset_name, model_tokenizer, template_config, tmvp_config, x)
    out["prompts"] = _VTC_PROMPT.format(question=out["question"])
    return out

  utils_rl.process_data = _patched_process_data
  print(
      "[my_train_rl] patched utils_rl.process_data ->"
      " raw VTC prompt (no chat template), ends with '<reasoning>\\n'",
      flush=True,
  )


# ---------------------------------------------------------------------
# 3. Tunix monkey-patches (numerical stability)
# ---------------------------------------------------------------------


def _patch_tunix_loss():
  """Cast loss to fp32 + clamp KL + ddof=0 advantages — VTC parity fixes."""
  import jax
  import jax.numpy as jnp
  import numpy as np

  from tunix.rl.grpo import grpo_learner as _gl
  from tunix.rl import common as _common

  _orig_grpo_loss_fn = _gl.grpo_loss_fn

  def _patched_grpo_loss_fn(model, train_example, algo_config, pad_id, eos_id):
    if hasattr(train_example, "advantages") and train_example.advantages is not None:
      train_example = train_example.replace(
          advantages=train_example.advantages.astype(jnp.float32)
      )
    loss, aux = _orig_grpo_loss_fn(
        model, train_example, algo_config, pad_id, eos_id
    )
    return loss.astype(jnp.float32), aux

  _gl.grpo_loss_fn = _patched_grpo_loss_fn

  _orig_kl = _common.compute_kl_divergence

  def _patched_kl(per_token_logps, ref_per_token_logps, method="mse_kl"):
    kl = _orig_kl(per_token_logps, ref_per_token_logps, method=method)
    kl = jnp.clip(kl.astype(jnp.float32), 0.0, 50.0)
    return kl

  _common.compute_kl_divergence = _patched_kl

  # Fix B: advantage normalization with ddof=0 + eps=1e-8 (matches VTC).
  # tunix default uses ddof=1 (sample std) and eps=1e-4.
  def _patched_compute_advantages(rewards, num_generations):
    rewards = np.asarray(rewards, dtype=np.float32)
    grouped = rewards.reshape(-1, num_generations)
    mean = grouped.mean(axis=-1).repeat(num_generations)
    std = grouped.std(axis=-1, ddof=0).repeat(num_generations)
    return (rewards - mean) / (std + 1e-8)

  _gl.compute_advantages = _patched_compute_advantages
  try:
    from tunix.rl import function_registry as _fr
    _fr.register_advantage_estimator("grpo")(_patched_compute_advantages)
  except Exception as e:
    print(f"[my_train_rl] note: function_registry re-register skipped: {e}", flush=True)

  print(
      "[my_train_rl] patched tunix:"
      " grpo_loss_fn (fp32),"
      " compute_kl_divergence (default=mse_kl, clip[0,50]),"
      " compute_advantages (ddof=0, eps=1e-8)",
      flush=True,
  )


# ---------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------


def _patch_vllm_rollout_kwargs():
  """Make tunix's VllmRollout.get_per_token_logps tolerate the
  `completion_mask` kwarg (and any other newer-tunix kwargs) that the
  trainer passes when num_iterations >= 2. This image's tunix lacks the
  parameter in the signature, causing TypeError and forcing num_iter=1
  (= inert PPO ratio = 1.0 = REINFORCE = no learning).

  Fixing this unblocks num_iter=2, which gives 2 real PPO updates per
  step (ratio ≠ 1 from bf16 drift on iter 1, ≠ 1 from policy movement
  on iter 2) — matches GPU NeMo-RL's effective ~2 mini-batch updates
  per step.
  """
  from tunix.rl.rollout import vllm_rollout
  _orig = vllm_rollout.VllmRollout.get_per_token_logps

  def _patched(self, prompt_tokens, completion_tokens, **kwargs):
    # silently absorb completion_mask + any other unknown kwargs
    return _orig(self, prompt_tokens, completion_tokens)

  vllm_rollout.VllmRollout.get_per_token_logps = _patched
  print(
      "[my_train_rl] patched VllmRollout.get_per_token_logps to absorb"
      " completion_mask kwarg → num_iterations=2 should now work",
      flush=True,
  )


def _patch_rollout_return_logprobs():
  """Force RolloutConfig.return_logprobs=True so vLLM populates per-token
  logprobs on the rollout output.

  Why: tunix's RolloutConfig.return_logprobs defaults to False
  (base_rollout.py:117), and MaxText's train_rl.py:504 doesn't override it.
  Consequence: vllm_sampler.py:530 passes `logprobs=None` to the vLLM
  engine, so `self.output.logprobs` ends up as a list of empty lists
  (verified empirically in v11 attempt #1: `len(rollout_logprobs)=32,
  first_seq_len=0`). Setting return_logprobs=True is the prerequisite for
  the `_patch_grpo_old_logprobs` patch below to have any data to read.
  """
  from tunix.rl.rollout import base_rollout as _br

  _orig_init = _br.RolloutConfig.__init__

  def _patched_init(self, *args, **kwargs):
    kwargs["return_logprobs"] = True
    _orig_init(self, *args, **kwargs)

  _br.RolloutConfig.__init__ = _patched_init
  print(
      "[my_train_rl] patched RolloutConfig.__init__ ->"
      " return_logprobs=True (prerequisite for old-logprobs-from-rollout)",
      flush=True,
  )


def _patch_grpo_old_logprobs():
  """Feed vLLM rollout-time logprobs as `old_per_token_logps` even with
  num_iterations=1, mirroring NeMo-RL's GPU path.

  Why: tunix's grpo_learner sets old_per_token_logps=None when
  num_iterations==1 (grpo_learner.py:296), and the loss falls back to
  jax.lax.stop_gradient(per_token_logps) (grpo_learner.py:514-515). That
  makes the PPO ratio exp(per_token_logps - old_per_token_logps) identically
  1.0, so the policy-gradient term collapses to -mean(advantage) ≈ 0 (group
  normalized) and only KL drives training. Direct evidence: actor/train/
  pg_clipfrac=0 uniformly across v6/v8/v9/v10 TBs.

  GPU NeMo-RL never takes that shortcut: even on the first PPO mini-batch,
  it feeds vLLM's rollout-time logprobs as old_logprobs. Bf16 drift between
  vLLM rollout and the Flax trainer forward gives probs_ratio ∈ ~[0.5, 2.0]
  (cross-checked from wandb run fqzrr3i7) → real PPO clipping → real
  policy-gradient signal.

  Implementation: VllmRollout caches its sampler output on self.output
  (vllm_rollout.py:93), so self.rl_cluster.rollout.output.logprobs holds
  the per-completion-token logprobs from the most recent generate() call.
  We pad them to match completion_mask.shape (right-pad with 0; masked out
  by completion_mask in the loss) and stamp them onto train_example.

  Sidesteps b/428730696 entirely: never invokes VllmRollout.get_per_token_logps
  (the broken path that crashes jnp.concatenate at rl_cluster.py:1034 when
  its Python list-of-lists return value hits jnp). We do the
  list→np.ndarray→jnp conversion ourselves, once per step, in driver code.
  """
  import jax.numpy as jnp
  import numpy as np
  from tunix.rl.grpo import grpo_learner as _gl

  _orig_gca = _gl.GrpoLearner._generate_and_compute_advantage
  _state = {"announced": False, "warned_missing": False}

  def _patched_gca(self, *args, **kwargs):
    train_example = _orig_gca(self, *args, **kwargs)
    if train_example.old_per_token_logps is not None:
      return train_example  # num_iter>=2 already populated; leave alone

    rollout_logprobs = getattr(
        getattr(self.rl_cluster.rollout, "output", None), "logprobs", None
    )
    if rollout_logprobs is None:
      if not _state["warned_missing"]:
        _state["warned_missing"] = True
        print(
            "[my_train_rl] WARN: rl_cluster.rollout.output.logprobs is None;"
            " falling back to tunix's stop_grad shortcut (probs_ratio≡1).",
            flush=True,
        )
      return train_example

    bs, max_len = train_example.completion_mask.shape
    padded = np.zeros((bs, max_len), dtype=np.float32)
    n_seqs = min(len(rollout_logprobs), bs)
    total_filled = 0
    for i in range(n_seqs):
      arr = np.asarray(rollout_logprobs[i], dtype=np.float32)
      n = min(arr.shape[0], max_len)
      padded[i, :n] = arr[:n]
      total_filled += n

    if total_filled == 0:
      # Empty rollout logprobs (e.g. return_logprobs=False). All-zero
      # old_per_token_logps would make ratio=exp(per_token_logps) which is
      # mathematically WORSE than the stop_grad shortcut (ratio≡1). Fall
      # back to leaving old_per_token_logps=None so tunix's loss takes
      # the safe stop_grad path.
      if not _state["warned_missing"]:
        _state["warned_missing"] = True
        print(
            "[my_train_rl] WARN: rollout_logprobs is all-empty"
            f" (len={len(rollout_logprobs)}, first_seq_len="
            f"{len(rollout_logprobs[0]) if len(rollout_logprobs) else 0})."
            " Likely RolloutConfig.return_logprobs=False. Falling back to"
            " stop_grad shortcut for this step (and all subsequent unless"
            " logprobs become populated).",
            flush=True,
        )
      return train_example

    if not _state["announced"]:
      _state["announced"] = True
      sample0 = rollout_logprobs[0]
      sample0_len = len(sample0) if hasattr(sample0, "__len__") else "?"
      print(
          "[my_train_rl] _generate_and_compute_advantage:"
          f" filling old_per_token_logps from rollout."
          f" completion_mask.shape={(bs, max_len)},"
          f" len(rollout_logprobs)={len(rollout_logprobs)},"
          f" first_seq_len={sample0_len},"
          f" element_type={type(sample0).__name__},"
          f" total_filled={total_filled},"
          f" mean_logp={float(padded[padded != 0].mean()) if total_filled else 'N/A'}",
          flush=True,
      )

    return train_example.replace(old_per_token_logps=jnp.asarray(padded))

  _gl.GrpoLearner._generate_and_compute_advantage = _patched_gca
  print(
      "[my_train_rl] patched GrpoLearner._generate_and_compute_advantage:"
      " old_per_token_logps now sourced from vLLM rollout (mimics NeMo-RL,"
      " brings probs_ratio ≠ 1 → real PPO clipping)",
      flush=True,
  )


def _patch_print_train_step():
  """Force-log every actor train step + eval step (the upstream peft_trainer
  Train-step log is silent in our build for unknown reasons; this gives us
  visible per-step signal so we can see progress / hangs)."""
  from tunix.rl import rl_cluster as _rc
  import time as _time

  _orig_update = _rc.RLCluster.update_actor
  _state = {"step": 0, "last_t": _time.time()}

  def _patched_update_actor(self, *args, **kwargs):
    _state["step"] += 1
    n = _state["step"]
    t0 = _time.time()
    dt_idle = t0 - _state["last_t"]
    out = _orig_update(self, *args, **kwargs)
    t1 = _time.time()
    _state["last_t"] = t1
    summary = ""
    if isinstance(out, dict):
      keys = sorted(out.keys())[:8]
      bits = []
      for k in keys:
        v = out[k]
        try:
          v = float(v)
          bits.append(f"{k}={v:.4g}")
        except Exception:
          pass
      summary = " ".join(bits)
    print(
        f"[ACTOR_STEP {n}] dt_idle={dt_idle:.1f}s step_dt={t1-t0:.1f}s {summary}",
        flush=True,
    )
    return out

  _rc.RLCluster.update_actor = _patched_update_actor
  print("[my_train_rl] patched RLCluster.update_actor for per-step prints", flush=True)


def _patch_pass_eval_ds_to_trainer():
  """Wire MaxText's `test_dataset` into tunix's `GRPOLearner.train` so
  `eval_interval` actually fires periodic evals.

  IMPORTANT: tunix's RL eval loop iterates the FULL eval_ds (and `eval_steps`
  config is SFT-only — doesn't apply here). At bs=4 prompts × 8 generations =
  32 vLLM seqs/batch × ~12 s, the full 330-batch test_dataset would be
  ~66 min per intermediate eval. So we slice eval_ds to a small subset
  (env `EVAL_INTERMEDIATE_BATCHES`, default 20 → ~80 prompts ~3-4 min/eval).
  """
  import os as _os
  from maxtext.trainers.post_train.rl import train_rl as _t
  from tunix.rl.grpo.grpo_learner import GrpoLearner as _GL

  _state = {"eval_ds": None}

  _orig_prepare = _t.prepare_datasets

  def _wrapped_prepare(*a, **kw):
    train_ds, test_ds = _orig_prepare(*a, **kw)
    intermediate_n = int(_os.environ.get("EVAL_INTERMEDIATE_BATCHES", "20"))

    # `test_ds` here is a grain IterDataset (already .to_iter_dataset()).
    # Slicing with `[:N]` doesn't actually limit it — the full dataset still
    # gets iterated. Wrap it in an iter-based take(N) adapter that tunix's
    # eval loop will actually respect (it does `iter(eval_ds)` then drains).

    class _TakeN:
      def __init__(self, src, n):
        self._src = src
        self._n = n

      def __iter__(self):
        i = 0
        for x in self._src:
          if i >= self._n:
            return
          i += 1
          yield x

    sliced = _TakeN(test_ds, intermediate_n)
    _state["eval_ds"] = sliced
    print(
        f"[my_train_rl] wrapped intermediate eval_ds in TakeN({intermediate_n})"
        f" so each intermediate eval is ~{intermediate_n} batches"
        f" (pre/post-eval go through the unwrapped test_ds, full 1319)",
        flush=True,
    )
    return train_ds, test_ds

  _t.prepare_datasets = _wrapped_prepare

  _orig_train = _GL.train

  def _train_with_eval(self, train_ds, eval_ds=None, skip_jit=False):
    if eval_ds is None and _state["eval_ds"] is not None:
      eval_ds = _state["eval_ds"]
      print(
          f"[my_train_rl] passing sliced eval_ds to GrpoLearner.train"
          f" so eval_interval={_os.environ.get('EVAL_INTERVAL', '?')}"
          f" fires intermediate evals on a small subset",
          flush=True,
      )
    return _orig_train(self, train_ds, eval_ds=eval_ds, skip_jit=skip_jit)

  _GL.train = _train_with_eval


def _patch_evaluate_rl():
  """Make evaluate_rl.score_responses use VTC's permissive `\\boxed{}`
  extraction (matches GPU repro), and report VTC mean reward.

  MaxText's default `score_responses` requires the model output to contain
  the full `<reasoning>...</reasoning><answer>...</answer>` structure to
  extract the answer. In practice, Qwen3-1.7B often produces correct
  numeric answers but with format quirks (e.g., `\\boxed{X}\\end{answer}`
  instead of `</answer>`, or runs over the token budget before closing
  the answer tag). Those answers get scored as wrong.

  VTC's eval scans for the last `\\boxed{...}` directly, which catches
  both cases. This patch swaps in that scorer.
  """
  from maxtext.trainers.post_train.rl import evaluate_rl as _ev
  from maxtext.utils import max_logging

  _orig_evaluate = _ev.evaluate

  def _rebatch(dataset, eval_bs):
    """Flatten the (batched) test dataset and re-yield in EVAL_BATCH_SIZE chunks."""
    bq, bp, ba = [], [], []
    for batch in dataset:
      n = len(batch["question"])
      for i in range(n):
        bq.append(batch["question"][i])
        bp.append(batch["prompts"][i])
        ba.append(batch["answer"][i])
        if len(bq) >= eval_bs:
          yield {"question": bq[:eval_bs], "prompts": bp[:eval_bs], "answer": ba[:eval_bs]}
          bq, bp, ba = bq[eval_bs:], bp[eval_bs:], ba[eval_bs:]
    if bq:
      yield {"question": bq, "prompts": bp, "answer": ba}

  def _vtc_evaluate(tmvp_config, dataset, rl_cluster, num_passes=1,
                    corr_lst=False, make_lst=False):
    import os as _os
    eval_bs = int(_os.environ.get("EVAL_BATCH_SIZE", "32"))
    print(f"[my_train_rl] eval rebatched to size {eval_bs}", flush=True)

    response_lst = []
    corr = 0
    partial = 0
    fmt = 0
    total = 0
    vtc_reward_sum = 0.0
    from tqdm.auto import tqdm
    for batch in tqdm(_rebatch(dataset, eval_bs)):
      answers = batch["answer"]
      questions = batch["question"]
      prompts = batch["prompts"]
      multi = _ev.generate_responses(
          tmvp_config=tmvp_config, prompts=prompts,
          rl_cluster=rl_cluster, num_passes=num_passes,
      )
      for question, responses, answer in zip(questions, multi, answers):
        response = responses[0] if responses else ""
        # VTC permissive scoring (also our reward fn's scorer).
        extract_err = None
        try:
          extracted = _extract_boxed(response)
        except Exception as ex:
          extracted = None
          extract_err = repr(ex)
        is_correct = (
            extracted is not None
            and str(extracted).replace(",", "").strip()
            == str(answer).replace(",", "").strip()
        )
        # Debug: dump first 5 (response_tail, gold, extracted) so we can see
        # exactly what the model's putting out and whether our extractor is wrong.
        if total < 5:
          tail = response[-400:] if len(response) > 400 else response
          print(
              f"\n[VTC-DBG #{total}] gold={answer!r} extracted={extracted!r}"
              f" is_correct={is_correct} extract_err={extract_err}\n"
              f"--- response tail (last 400 chars) ---\n{tail}\n--- end ---\n",
              flush=True,
          )
        # Format check (ours and VTC's): one </reasoning>, one <answer> + one </answer>, ordered
        has_r = response.count("</reasoning>") == 1
        has_a = (
            response.count("<answer>") == 1
            and response.count("</answer>") == 1
        )
        ir = response.find("</reasoning>")
        iao = response.find("<answer>")
        iac = response.find("</answer>")
        ordered = has_r and has_a and (ir < iao < iac)
        fmt_ok = has_r and has_a and ordered
        if fmt_ok:
          fmt += 1
        if is_correct:
          corr += 1
          partial += 1
        # VTC reward
        if not fmt_ok:
          r = 0.5 if is_correct else 0.0
        else:
          r = 1.0 if is_correct else 0.1
        vtc_reward_sum += r
        total += 1
        if total % 50 == 0:
          max_logging.log(
              f"===> {corr=}, {total=}, accuracy={corr/total*100:.2f}%, "
              f"format={fmt/total*100:.2f}%, "
              f"vtc_mean_reward={vtc_reward_sum/total:.4f}"
          )
    max_logging.log(
        f"VTC EVAL: corr={corr}/{total}, accuracy={corr/total*100:.2f}%, "
        f"format={fmt/total*100:.2f}%, vtc_mean_reward={vtc_reward_sum/total:.4f}"
    )
    return (corr, total, corr / total * 100 if total else 0,
            partial / total * 100 if total else 0,
            fmt / total * 100 if total else 0), response_lst

  _ev.evaluate = _vtc_evaluate
  print(
      "[my_train_rl] patched evaluate_rl.evaluate ->"
      " VTC permissive `\\boxed{}` extraction + vtc_mean_reward",
      flush=True,
  )


def main(argv: list[str]) -> None:
  import os as _os

  _patch_maxtext_rewards()
  _patch_process_data()
  _patch_tunix_loss()
  _patch_evaluate_rl()
  _patch_pass_eval_ds_to_trainer()
  _patch_vllm_rollout_kwargs()

  # v11 mechanism patches — gated behind ENABLE_ROLLOUT_OLDLOGPS=1 (default off).
  # When enabled, vLLM is asked to return per-token logprobs and those are stamped
  # onto train_example.old_per_token_logps so PPO ratio ≠ 1 (real clipping).
  # Cost: ~+50% per-step wall (return_logprobs=True slows the rollout).
  # Benefit (measured v11 vs v10, same hyperparams): pg_clipfrac 0 → ~10%, but
  # post-RL VTC only +0.79pp. Default off so future experiments stay fast; flip
  # on for mechanism studies. See scratchpad_maxtext.md "v11" section.
  if _os.environ.get("ENABLE_ROLLOUT_OLDLOGPS", "0") == "1":
    _patch_rollout_return_logprobs()
    _patch_grpo_old_logprobs()
  else:
    print(
        "[my_train_rl] ENABLE_ROLLOUT_OLDLOGPS=0 (default):"
        " skipping rollout-old-logprobs patches; using tunix's stop_grad"
        " shortcut (probs_ratio≡1, pg_clipfrac=0). Set"
        " ENABLE_ROLLOUT_OLDLOGPS=1 to enable real PPO clipping at +50% step time.",
        flush=True,
    )

  _patch_print_train_step()

  # Defer the import until after patching so MaxText sees patched modules.
  from maxtext.trainers.post_train.rl import train_rl as _train_rl

  _train_rl.main(argv)


if __name__ == "__main__":
  from absl import app
  app.run(main)
