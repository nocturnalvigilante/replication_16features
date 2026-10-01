#!/bin/bash
#SBATCH --job-name=main_recheck
#SBATCH --output=incremental_experiments/results/main_recheck/slurm_%A_task%a.log
#SBATCH --cpus-per-task=4
#SBATCH --partition=threedle-own,threedle-contrib,general,peanut-cpu
#SBATCH --requeue
#SBATCH --open-mode=append

# Recheck of the myopic weakness tables through Kanix's own main(), not our
# sweep drivers. Nothing here computes anything: it only calls his CLI.
#
#   tasks 0-27  2-gene: his UNMODIFIED scripts/search_multigene_myopic_vs_stop.py,
#               one task per (family, preset, allele) = all 420 configs.
#   tasks 28-39 3-gene: scripts/search_multigene_myopic_vs_stop_3gene.py (his
#               script + GeneC). His main() has no GeneC flag, so get_config
#               gives GeneC its default 0.1 -- which is exactly our MixedA GeneC
#               (0.02, 0.10, 0.10). So the 60 MixedA 3-gene configs go through
#               his main() untouched. 28-29 ThreeGeneration, 30-39 Extended
#               split by fixed cost.
#
# pulp is not importable in genetic-rl, so his solve_exact_dual_pulp falls back
# to solve_exact_dp_primal (backward induction, the ground truth), and his
# extract_exact_policy gives the DP root action.
#
#   sbatch --array=0-27  --mem=32G      --time=01:00:00 submit_main_recheck.sh
#   sbatch --array=28-29 --mem=48G      --time=01:00:00 submit_main_recheck.sh
#   sbatch --array=30-39 --mem=215040   --time=02:30:00 submit_main_recheck.sh
# Compare: python incremental_experiments/main_recheck_compare.py

cd /net/projects/ranalab/rajhansini/replication_16features
PY=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
OUT=incremental_experiments/results/main_recheck
mkdir -p "$OUT"

FAMS=(ThreeGeneration Extended); PRES=(Base Aggressive)
AL=(UltraRare VeryRare LowLow LowHigh MixedA MediumEven HighHigh)
AF_A=(0.005 0.01 0.02 0.02 0.02 0.08 0.15)
AF_B=(0.005 0.01 0.02 0.15 0.10 0.08 0.15)
FIXED=(0.005 0.01 0.015 0.02 0.03)
T=$SLURM_ARRAY_TASK_ID

if [ "$T" -lt 28 ]; then
  f=$((T / 14)); p=$(((T / 7) % 2)); a=$((T % 7))
  SCRIPT=scripts/search_multigene_myopic_vs_stop.py; FC="${FIXED[*]}"
  TAG=g2_${FAMS[$f]}_${PRES[$p]}_${AL[$a]}
elif [ "$T" -lt 30 ]; then
  f=0; p=$((T - 28)); a=4
  SCRIPT=scripts/search_multigene_myopic_vs_stop_3gene.py; FC="${FIXED[*]}"
  TAG=g3_${FAMS[$f]}_${PRES[$p]}_${AL[$a]}
else
  i=$((T - 30)); f=1; p=$((i / 5)); a=4
  SCRIPT=scripts/search_multigene_myopic_vs_stop_3gene.py; FC="${FIXED[$((i % 5))]}"
  TAG=g3_${FAMS[$f]}_${PRES[$p]}_${AL[$a]}_f${FC}
fi

echo "task=$T tag=$TAG script=$SCRIPT host=$(hostname) start=$(date -Is)"
$PY -c "import pulp" 2>/dev/null && { echo "pulp importable -- would run his LP, abort"; exit 1; }
$PY -u "$SCRIPT" --families "${FAMS[$f]}" --presets "${PRES[$p]}" \
  --af-a-values "${AF_A[$a]}" --af-b-values "${AF_B[$a]}" \
  --fixed-cost-values $FC --variable-cost-values 0.01 0.02 0.03 \
  --output-json "$OUT/$TAG.json" --output-md "$OUT/$TAG.md"
echo "end=$(date -Is)"
