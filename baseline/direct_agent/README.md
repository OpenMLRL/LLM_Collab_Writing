# Direct single-agent RLHF / DPO

One Qwen3-8B author writes both TLDR paragraphs directly. This is not the cen
coordinator protocol. Existing entrypoints and CoMLRL trainers are unchanged.

Run `python -m baseline.direct_agent.run --algorithm rlhf --seed 42 --output-dir OUTPUT`
or use `dpo`. `--dry-run` validates the pinned dataset without GPU weights.
The deployment should supply `--config` with a frozen local data export.

- Data: pinned `trl-lib/tldr`, first 250 train / first 250 test. No annotated
  completion is retained. Each algorithm uses identical data and seeded sampling.
- Output: concise paragraph, one blank line, expanded paragraph. Exactly two
  natural paragraphs (or the legacy `[PARAGRAPH_SPLIT]` delimiter) are accepted;
  there is no arbitrary text slicing. Score is existing structural/style
  reward / 3, bounded [0,1]; it is **not** a factuality or semantic-quality metric.
- Full bf16 Qwen3-8B, SDPA, non-reentrant checkpointing, no LoRA or quantization.
- C80/P16: 80 initial-model candidates/question, at most 16 largest-gap strict
  pairs. Ties are excluded. Decoding microbatch 20; max 512 new tokens for the
  complete two-paragraph response; temperature .7, top-p .8, top-k 20.
- DPO: beta .1, fixed initial reference, one epoch, at most 8000 response
  presentations. Reference logps are cached exactly before updating the actor.
- RLHF: one offline preference collection; full 8B scalar reward model fitted
  for one epoch, then frozen; GRPO actor with G4, eight epochs (8000 online
  responses). Synthetic evaluator preferences, not human feedback. No iteration
  replay or model comparator. Learned RM score is unbounded; true task reward
  remains [0,1]. Actor LR2e-5, RM LR1e-5, no reference KL.
- Separate backward graphs implement the exact pairwise RM/DPO gradient.
- W&B: OpenMLRL/TLDR, `tldr-direct-{rlhf|dpo}-qwen3-8b-seed{seed}-{job}`.
  `eval/turn_1/*` uses fixed first-16 anchors every320 actor responses, including
  initial/final chart points. Full250 initial/final metrics are saved locally
  and in run summary. Offline collection and RM fitting have their own axes.
  No train panel, eval_full panel, sample tables or forced environment-step chart.
- `python -m baseline.direct_agent.aggregate OUTPUT1 OUTPUT2 OUTPUT3 OUTPUT4`
  requires both methods with seeds42/43 complete; reports full-eval mean,
  sample SD and within-run change from initial. No partially finished runs.

RLHF typically needs two ~95GiB GPUs (actor + frozen RM); DPO one, subject to
the per-deployment pressure test. Source snapshots/run scripts are outside Git;
new source is left for local review and is not automatically committed.
