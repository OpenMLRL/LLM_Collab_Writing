# Long single-agent RLHF runs

Use the opt-in `long_run` entrypoint with `--algorithm rlhf --target-updates 3200
--segment-seconds 36000 --checkpoint-seconds 3600 --resume`. The short entrypoint,
team trainers, prompts, task scores, C/P selection and learning rates are unchanged.

RLHF retains the original two-stage method: task-scored strict preference pairs
fit a scalar reward model with the pairwise logistic loss; the frozen learned
reward supplies per-question GRPO advantages. It is not PPO or IAC.
`turn_1/reward_mean` and `eval/*` measure task reward; `training_reward_mean`
is the learned scalar (different scale). The 3200 target counts actor optimizer
updates only, not collected responses or reward-model steps.

`rlhf_setup.pt` atomically persists each completed sampling question, or the
reward-model weights/head, Adam state, shuffled pair cursor and RNG during fitting.
After fitting it stores the ready frozen RM and is reused on actor resumption.
`checkpoint.pt` contains actor weights, Adam, shuffled question cursor and RNG.
The output directory's identity fixes the configuration, data hashes and budget.
Use the same output directory and CUDA device count for continuation.

SIGUSR1 or the soft deadline pauses only at a durable question/optimizer boundary
and exits 85. Other exceptions fail normally and must not be blindly requeued.
An external 12-hour Slurm launcher should verify `checkpoint_status.json`,
check that `work_progress` advances, and requeue the same job only on exit 85.
Use a 10-hour soft deadline plus SIGUSR1 30 minutes before the hard limit.
Initial/final full evaluation is not interrupted; leave enough margin for it.

Regression tests compare original RLHF trajectories and exact CPU continuation
across preference collection, RM fitting and actor updates; they also cover the
existing direct algorithms. Changing hardware can cause floating-point variation.

