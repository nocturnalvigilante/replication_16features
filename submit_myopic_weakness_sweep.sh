#!/bin/bash
#SBATCH --job-name=myo_weak
#SBATCH --output=incremental_experiments/results/slurm_myoweak_%A_task%a.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-419
#SBATCH --requeue
#SBATCH --open-mode=append

# Map where myopic breaks: allele rarity x testing cost, on Kanix's families
# and his solver. 7 allele pairs x 5 fixed_cost x 3 variable_cost x 2 families
# x 2 presets = 420 rows. Costs run past his 0.015 ceiling, alleles below his
# 0.02 floor, so both axes can be seen saturating (or not).
#
# One row per task: own JSON + skip guard, so the array is safe to resubmit and
# only redoes what is missing. Rows are cheap (Extended 2-gene = 107,728 states,
# seconds), so 2h is generous.

cd /net/projects/ranalab/rajhansini/replication_16features
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python -u \
  incremental_experiments/myopic_weakness_sweep.py --row "$SLURM_ARRAY_TASK_ID"
