#!/bin/bash
#SBATCH --job-name=myo_kanix102
#SBATCH --output=incremental_experiments/results/slurm_myokanix_%A_task%a.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-101
#SBATCH --requeue
#SBATCH --open-mode=append

# Myopic baseline over Kanix's OWN 102-family suite (original8 + local40 +
# phase6_54). His full102_expected.json has ADP ratio2/ratio3 only -- myopic was
# never his baseline -- so this fills that column on exactly his settings.
#
# One row per task so memory is released between configs (verify_myopic_one.py
# documents a single long-lived process growing past 16GB). Each task writes its
# own JSON behind a skip guard, so the array is safe to resubmit and only redoes
# what is missing. Rows are cheap: Extended 2-gene is 107,728 states and solves
# in seconds, so 2h is generous.

cd /net/projects/ranalab/rajhansini/replication_16features
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python -u \
  incremental_experiments/myopic_on_kanix_suite.py --row "$SLURM_ARRAY_TASK_ID"
