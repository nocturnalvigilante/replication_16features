#!/bin/bash
#SBATCH --job-name=e6_cost
#SBATCH --output=incremental_experiments/results/slurm_e6cost_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# E6 (2-gene) on the 83 high-cost cells where myopic is weak.
#
#   tasks 0-2  --mode eval   existing E6 checkpoints (trained at f=0.01 only),
#                            scored as-is. Does what E6 learned transfer?
#   tasks 3-5  --mode train  retrain E6 with fixed_cost VARYING in TRAIN
#                            (0.010/0.020/0.030), family split untouched.
#
# Both scored on the identical 83 cells, read straight from the myopic sweep's
# SUMMARY.json, so GNN and myopic numbers line up cell for cell.
#
# Resume: train mode checkpoints every epoch and picks up where it left off;
# a finished (mode,seed) skips in <1s via the results.json guard, so this array
# is safe to resubmit. USR1 300s before the wall requeues rather than dying.

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  MODE=eval;  SEED=$SLURM_ARRAY_TASK_ID
else
  MODE=train; SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi
echo "task=$SLURM_ARRAY_TASK_ID -> mode=$MODE seed=$SEED"

requeue_handler() {
    echo "[SIGNAL] $(date -Is) USR1 (300s to wall) -- requeueing task $SLURM_ARRAY_TASK_ID"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python -u \
  incremental_experiments/e6_cost_experiment.py --mode "$MODE" --seed "$SEED" --epochs 500 &
CHILD=$!
wait "$CHILD"
