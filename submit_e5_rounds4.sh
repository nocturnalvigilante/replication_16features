#!/bin/bash
#SBATCH --job-name=e5_rounds4
#SBATCH --output=incremental_experiments/results/slurm_e5r4_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --open-mode=append

# E5 at n_rounds=4 -- fills the gaps in the message-passing round sweep.
#
# Why 4: undirected pedigree diameter is 2 hops (Trio, Nuclear), 3 (ThreeGeneration)
# and 4 (Extended). n_rounds=2 -- what E4/E6/E7/E9 all use -- cannot move
# information across ThreeGeneration or Extended at all. The existing sweep already
# shows 2 -> 3 is worth 42% on 2-gene ThreeGeneration (0.1782 -> 0.1038, seed 0),
# and n_rounds=3 is the best 2-gene result on record (0.0959, 3 seeds).
# 4 is the count Extended's geometry actually needs, and it is currently seed-0-only
# at 2-gene and entirely absent at 3-gene.
#
# 5 tasks: 2-gene seeds 1,2 (seed 0 already done) + 3-gene seeds 0,1,2.

cd /net/projects/ranalab/rajhansini/replication_16features
PY=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python

case $SLURM_ARRAY_TASK_ID in
  0) SCRIPT=incremental_experiments/e5_train_two_gene_gnn_q.py; SEED=1 ;;
  1) SCRIPT=incremental_experiments/e5_train_two_gene_gnn_q.py; SEED=2 ;;
  2) SCRIPT=incremental_experiments/e5_train_gnn_q.py;          SEED=0 ;;
  3) SCRIPT=incremental_experiments/e5_train_gnn_q.py;          SEED=1 ;;
  4) SCRIPT=incremental_experiments/e5_train_gnn_q.py;          SEED=2 ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID -> $SCRIPT seed=$SEED n_rounds=4"
$PY -u "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED" --n_rounds 4
