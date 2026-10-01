#!/bin/bash
#SBATCH --job-name=e5_weak
#SBATCH --output=incremental_experiments/results/e5_weak/slurm_%A_task%a.log
#SBATCH --cpus-per-task=2
#SBATCH --partition=threedle-own,threedle-contrib,general,peanut-cpu
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# E5 GNN-Q (n_rounds=3) trained on the myopic-weakness grid -- incremental_experiments/e5_weak.py.
# GNN only (no MLP). TRAIN Trio+Nuclear x 420 settings, TEST the sweep's 420
# ThreeGeneration+Extended configs, per gene count.
#   submit_e5_weak.sh STAGE GENES
#     data   array 0-1  -> Trio, Nuclear
#     train  array 0-2  -> seed (500 epochs, checkpoint every epoch)
#     eval   array 0-27 -> unit: 0-13 ThreeGeneration, 14-27 Extended (15 configs each)
# Every stage skips finished work and USR1 300s before the wall requeues the
# task, so a cut-off task resumes (train from its last epoch).
#
#   D2=$(sbatch --parsable --array=0-1 --mem=16G --time=01:00:00 submit_e5_weak.sh data 2)
#   D3=$(sbatch --parsable --array=0-1 --mem=32G --time=02:00:00 submit_e5_weak.sh data 3)
#   G2=$(sbatch --parsable -d afterok:$D2 --array=0-2 --mem=16G --time=04:00:00 submit_e5_weak.sh train 2)
#   G3=$(sbatch --parsable -d afterok:$D3 --array=0-2 --mem=32G --time=04:00:00 submit_e5_weak.sh train 3)
#   sbatch -d afterok:$G2 --array=0-27  --mem=32G    --time=02:00:00 submit_e5_weak.sh eval 2
#   sbatch -d afterok:$G3 --array=0-13  --mem=64G    --time=04:00:00 submit_e5_weak.sh eval 3
#   sbatch -d afterok:$G3 --array=14-27 --mem=215040 --time=04:00:00 submit_e5_weak.sh eval 3   # 9,190,992 states
# Then: python incremental_experiments/e5_weak.py --stage summarize --genes 2   (and 3)

STAGE="$1"; GENES="$2"
cd /net/projects/ranalab/rajhansini/replication_16features
# One thread: on these tiny batches torch is ~4x faster at 1 thread than at 4
# (smoke test, one 3-gene Nuclear epoch: 1.6s vs 6.9s).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=$SLURM_ARRAY_TASK_ID
case "$STAGE" in
  data)  ARGS=(--family "$([ "$T" = 0 ] && echo Trio || echo Nuclear)") ;;
  train) ARGS=(--model gnn --seed "$T") ;;
  eval|traj)  ARGS=(--unit "$T") ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "task=$T stage=$STAGE genes=$GENES ${ARGS[*]} ${*:3} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"

requeue_handler() {
    echo "[SIGNAL] $(date -Is) USR1 (300s to wall) -- requeueing task $T"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python -u \
  incremental_experiments/e5_weak.py --stage "$STAGE" --genes "$GENES" "${ARGS[@]}" "${@:3}" &
CHILD=$!
wait "$CHILD"
echo "end=$(date -Is)"
