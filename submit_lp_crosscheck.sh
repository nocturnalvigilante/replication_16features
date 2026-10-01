#!/bin/bash
#SBATCH --job-name=lp_xcheck
#SBATCH --output=incremental_experiments/results/lp_crosscheck/slurm_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Does Kanix's LP exact DP agree with the backward-induction exact DP every
# sweep used? PuLP is not installed in genetic-rl, so his solve_exact_dual_pulp
# has always silently fallen back to solve_exact_dp_primal. Here PuLP is put on
# PYTHONPATH from an isolated folder (shared env untouched) and his UNMODIFIED
# scripts/search_multigene_myopic_vs_stop.py main() is run, so exact DP goes
# through his LP (Gurobi, his default backend). Each Gurobi log is kept as proof
# the LP actually ran. Sequential: the Gurobi WLS license allows 2 sessions.
#
# Five 2-gene configs = myopic_weakness_sweep rows 45 / 74 / 205 / 284 / 320.
# Compare with: python incremental_experiments/lp_crosscheck_compare.py

cd /net/projects/ranalab/rajhansini/replication_16features
export PYTHONPATH=/net/projects/ranalab/rajhansini/pylibs_pulp
export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic
PY=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
OUT=incremental_experiments/results/lp_crosscheck

$PY -c "import pulp; print('pulp', pulp.__version__, pulp.__file__, pulp.listSolvers(onlyAvailable=True))" || exit 1

run() {  # tag family preset af_a af_b fixed variable
  echo "=== $1 $(date -Is)"
  EXACT_DUAL_LOG_PATH=$OUT/$1_gurobi.log $PY -u scripts/search_multigene_myopic_vs_stop.py \
    --families "$2" --presets "$3" --af-a-values "$4" --af-b-values "$5" \
    --fixed-cost-values "$6" --variable-cost-values "$7" \
    --output-json $OUT/$1.json --output-md $OUT/$1.md
}
run row0045 ThreeGeneration Base       0.02  0.15  0.005 0.01
run row0074 ThreeGeneration Base       0.02  0.10  0.03  0.03
run row0205 ThreeGeneration Aggressive 0.15  0.15  0.02  0.02
run row0284 Extended        Base       0.02  0.10  0.03  0.03
run row0320 Extended        Aggressive 0.005 0.005 0.01  0.03
echo "=== done $(date -Is)"
