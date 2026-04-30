# Qwen3-1.7B GSM8K GRPO on TPU v6e — VTC-faithful recipe

This branch is an attempt to reproduce the GPU NeMo-RL GRPO recipe for
Qwen3-1.7B on GSM8K (which hits ~81% mean VTC partial-credit reward
in 200 steps) on TPU v6e using MaxText + tunix.

**Status (as of this commit):** Pipeline runs end-to-end. Pre-RL VTC mean
reward matches GPU within 1.1 pp (0.5450 vs GPU 0.5600). Post-RL is the
same as pre-RL (≈ 0.54), because tunix on TPU has a known limitation
that forces `num_iterations=1`, which makes the GRPO update degenerate
into REINFORCE with mean-zero advantages — i.e. the loss has no real
gradient signal and the policy doesn't move. See `# Open issue` below.

This branch is meant as a **reproducible starting point** for the TPU
team to iterate on. All monkey-patches are isolated in `my_train_rl.py`,
not scattered across tunix/MaxText source.

## Pin

This branch is based on `maxtext-v0.2.1` (commit `61fa4f38`), which is
exactly the commit baked into the Docker image
`gcr.io/infinipod-shared-dev/maxtext_post_training_stable:0.2.1`.
We md5-verified that 7 key files in the image match this commit
byte-for-byte; the only image-side drift is a sed-patch to
`globals.py` adding `qwen3-1.7b` HF tokenizer mapping, which we
reproduce here as a real edit.

## What's in this branch (delta vs `maxtext-v0.2.1`)

| File | What changed |
|---|---|
| `src/maxtext/utils/globals.py` | added `"qwen3-1.7b": "Qwen/Qwen3-1.7B"` and `qwen3-1.7b-base` to `HF_IDS` |
| `my_train_rl.py` | NEW — wrapper around `maxtext.trainers.post_train.rl.train_rl.main` that monkey-patches tunix + MaxText (see "Patches" below) |
| `run_qwen3_1p7b_vtc.sh` | NEW — launcher that wraps `my_train_rl.py` with the GPU-parity config knobs and exposes everything as env vars |
| `pathways_job_qwen3_1p7b_vtc.yaml` | NEW — k8s `PathwaysJob` manifest for an idle controller pod (sleep infinity); intended for `kubectl apply` then `kubectl exec` workflow |
| `REPRO_QWEN3_1P7B_VTC_TPU.md` | this file |

## Patches in `my_train_rl.py`

All patches are applied at startup, before MaxText's `train_rl.main`
imports. They wrap upstream code without modifying it. In order:

1. **Reward function override** — replaces MaxText's three default reward
   functions (`match_format_exactly`, `match_format_approximately`,
   `check_numbers`) with a single VTC partial-credit reward
   (0/0.1/0.5/1.0) that mirrors GPU NeMo-RL's `VTCMathVerifyWorker`.

2. **Raw VTC prompt** — overrides `utils_rl.process_data` so the prompt
   becomes a raw text string ending with `<reasoning>\n` (no
   `apply_chat_template`). This is critical: with chat-template wrapping,
   Qwen3 triggers its native `<think>` tag mode and ignores our
   `<reasoning>...</reasoning>` instructions, dropping pre-RL accuracy
   by ~20 pp.

3. **fp32 GRPO loss + KL clamp** — patches `tunix.rl.grpo.grpo_learner.grpo_loss_fn`
   to cast advantages to fp32, and `tunix.rl.common.compute_kl_divergence`
   to default to `mse_kl` (= GPU's "k2") and clamp output to `[0, 50]`.

4. **Advantage normalization** — re-registers tunix's `compute_advantages`
   with `ddof=0` and `eps=1e-8` (matching the VTC source); the tunix
   default is `ddof=1, eps=1e-4`.

5. **Permissive `\boxed{}` extraction in eval** — replaces MaxText's
   `evaluate_rl.evaluate` with one that uses the VTC `_extract_boxed`
   regex (catches answers in `\boxed{N}` whether or not they're wrapped
   in `<answer>...</answer>` — important because the model often emits
   `\boxed{N}\end{answer}` instead of `</answer>`). Also reports
   `vtc_mean_reward` (the GPU-comparable metric) alongside accuracy.

6. **eval re-batching to size 32** — wraps the test dataset iterator so
   pre/post-eval runs at vLLM batch=32 (~5× faster) while training rollouts
   stay at batch=4 to preserve GPU-parity dynamics. Reads `EVAL_BATCH_SIZE`
   env var.

7. **eval_ds wiring + slicing** — passes the test dataset to
   `GrpoLearner.train` (MaxText's `train_rl` doesn't, so tunix's
   `eval_every_n_steps` is dormant by default). Wraps eval_ds in a
   `TakeN(20)` adapter so intermediate evals at step 50/100/150 take
   ~3-4 min instead of ~66 min (tunix's RL eval ignores `eval_steps` config
   and iterates the full dataset otherwise). Reads `EVAL_INTERMEDIATE_BATCHES`
   env var.

8. **`completion_mask` kwarg absorber** — patches
   `tunix.rl.rollout.vllm_rollout.VllmRollout.get_per_token_logps`
   to silently absorb the `completion_mask` kwarg the trainer passes when
   `num_iterations >= 2`. Was needed to even *attempt* `num_iter=2`, but
   see "Open issue" below — that path hits a deeper bug.

9. **Per-step train log** — patches `tunix.rl.rl_cluster.RLCluster.update_actor`
   to print `[ACTOR_STEP N]` with timing on every train step. Useful for
   "is training making progress?" health checks since the upstream
   `peft_trainer` log line wasn't always firing in our setup.

## How to reproduce

### 1. Apply the PathwaysJob

```bash
kubectl apply -f pathways_job_qwen3_1p7b_vtc.yaml
```

The pod sleeps until you kubectl-cp the maxtext repo in (it waits for
`/workspace/tunix/pyproject.toml`). Wait ~2 min for the controller pod
+ TPU workers to reach Running.

### 2. Sync this maxtext branch into the pod

```bash
POD=$(kubectl get pods -o name | grep qw3-17b-rl-pathways-head | head -1 | sed s,pod/,,)
kubectl exec $POD -c main -- bash -c '[ -d /deps/src/maxtext ] && mv /deps/src/maxtext /deps/src/maxtext.disabled || true'
kubectl cp /path/to/this/maxtext $POD:/workspace/ -c main
kubectl exec $POD -c main -- bash -c 'cd /workspace/maxtext && pip install -e . --no-deps'
```

(We disable `/deps/src/maxtext` so our copy wins the namespace-package
import lookup. Local copy is then editable in-place.)

### 3. Launch a run

```bash
kubectl exec $POD -c main -- bash -c '
  NUM_BATCHES=150 \
  EVAL_INTERVAL=50 \
  EVAL_INTERMEDIATE_BATCHES=20 \
  EVAL_BATCH_SIZE=32 \
  RUN_NAME=qwen3-1.7b-grpo-vtc-runX \
  nohup bash /workspace/maxtext/run_qwen3_1p7b_vtc.sh \
    > /workspace/runX.log 2>&1 &
'
```

Wall-clock: ~9 min pre-eval + ~30 min train + ~3 min × 4 intermediate evals
+ ~9 min post-eval ≈ **~60-80 min** total.

Tensorboard:
`gs://vtc-pathways-scratch-infinipod-shared-dev/runs/qwen3-1.7b-grpo-vtc/${RUN_NAME}/tensorboard/`

All training-relevant knobs are env-overridable (see top of
`run_qwen3_1p7b_vtc.sh`): `NUM_BATCHES`, `LR`, `GRPO_BETA`,
`GRPO_EPSILON`, `NUM_GENERATIONS`, `NUM_ITERATIONS`, `ROLLOUT_TEMP`, etc.
Defaults are the GPU recipe values.

## Open issue (the reason post-RL ≈ pre-RL)

`num_iterations=2` is the *intended* GRPO setting (matches GPU NeMo-RL's
~2 mini-batch updates per step), but tunix's TPU vLLM integration has
two bugs along that code path:

1. `VllmRollout.get_per_token_logps()` doesn't accept `completion_mask`
   kwarg → easy fix (patch #8 above).
2. `tunix/rl/rl_cluster.py:993` does `jnp.concatenate(outs, axis=0)` on
   what is actually a list-of-lists when vLLM returns logprobs on TPU
   (tracked as Google internal bug `b/428730696`; vllm_rollout.py's own
   comment says *"we cannot return self.output.logprobs yet"*). This bug
   is fundamental — without real per-token logprobs from vLLM, tunix
   cannot do `num_iterations >= 2` on TPU.

Result: tunix on TPU is forced to `num_iterations=1`, where the loss takes
a `stop_grad` shortcut making `ratio = 1.0` for every token. PPO clipping
is inert; loss reduces to `-mean(advantage) + β·KL` which has expected
value ≈ 0 (advantages are group-normalized to mean 0). The actual
gradient is REINFORCE-with-baseline (non-zero in principle), but combined
with the GPU recipe's tiny LR (2e-7) and the KL pulling back to π_ref,
the policy effectively doesn't move over 150 steps.

GPU NeMo-RL avoids this because it always feeds vLLM rollout-time logprobs
in as `old_logprobs` (not the `stop_grad` shortcut), so `ratio ≠ 1` from
the very first update of every step due to bf16 numerical drift between
vLLM and the trainer's kernel.

### Fixes the TPU team can try

- **Fix `b/428730696`** in tunix/vLLM-TPU integration so vLLM logprobs
  come back as a JAX array instead of a Python list. Then `num_iter=2`
  works, ratio is real, learning kicks in.
- **Bypass the shortcut** in `num_iter=1` mode: monkey-patch
  `tunix.rl.grpo.grpo_learner.grpo_loss_fn` to populate
  `train_example.old_per_token_logps` from the rollout's logprobs (when
  available) instead of falling through to `stop_grad(per_token_logps)`.
- **Brute force**: bump LR by 10-100× (we're testing this in run-v8).
- **Drop KL** (β=0) to remove the pull-back force.

## Files

```
maxtext-v0.2.1 + delta
├── src/maxtext/utils/globals.py           # qwen3-1.7b registration
├── my_train_rl.py                         # all monkey-patches
├── run_qwen3_1p7b_vtc.sh                  # launcher
├── pathways_job_qwen3_1p7b_vtc.yaml       # k8s manifest
└── REPRO_QWEN3_1P7B_VTC_TPU.md            # this file
```

