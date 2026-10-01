#!/bin/bash
#SBATCH --job-name=ext_e5
#SBATCH --output=incremental_experiments/results/slurm_ext_e5_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# Extended-family (OOD) four-way logs for the E5 variant (E4 architecture,
# 3 message-passing rounds), BOTH gene counts, 3 seeds -- 6 tasks.
# Tasks 0-2 = 3-gene seeds 0-2, tasks 3-5 = 2-gene seeds 0-2.
#
# Same resume machinery as submit_extended_fourway_e6e7_rerun.sh: 3-gene
# Extended solves exact DP over 9,190,992 states per config with no cache, so
# a seed needs several 4h slots. Each config is checkpointed to
# seed*/partial_e5/ when solved; USR1 before the wall requeues the task and it
# skips what it already has. Partials are keyed to the SLURM job id, so a fresh
# sbatch wipes them rather than inheriting them.

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then GENES=3; SEED=$SLURM_ARRAY_TASK_ID; else GENES=2; SEED=$(( SLURM_ARRAY_TASK_ID - 3 )); fi

echo "task=$SLURM_ARRAY_TASK_ID -> variant=e5 genes=$GENES seed=$SEED"

requeue_handler() {
    echo "[SIGNAL] $(date -Is) USR1 received (300s to wall clock) -- requeueing task ${SLURM_ARRAY_TASK_ID}; it will resume from its solved configs"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_extended_fourway.py --genes "$GENES" --seed "$SEED" \
  --variant e5 --device cpu &
CHILD=$!
wait "$CHILD"
