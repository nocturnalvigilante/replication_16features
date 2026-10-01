#!/bin/bash
#SBATCH --job-name=e5w_monitor
#SBATCH --output=incremental_experiments/results/e5_weak/monitor_slurm_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --partition=threedle-own,threedle-contrib,general,peanut-cpu

# Overnight watchdog for the E5-weak pipeline: one pass of
# incremental_experiments/e5_weak_monitor.py every 5 minutes (resubmits
# timed-out / failed / OOM'd work, alerts on guard failures, summarizes at the end).
# It queues its own successor (afterany) the moment it starts, so the watch
# survives its 4h wall, a node failure or a kill. Stops once results/e5_weak/DONE exists.
#   sbatch submit_e5_weak_monitor.sh          status: results/e5_weak/STATUS.txt

cd /net/projects/ranalab/rajhansini/replication_16features
R=incremental_experiments/results/e5_weak
PY=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
[ -f "$R/DONE" ] && exit 0
OTHER=$(squeue -h -u "$USER" -n e5w_monitor -t RUNNING -o %A | grep -v "^${SLURM_JOB_ID}$")
if [ -n "$OTHER" ]; then echo "another monitor is running ($OTHER); exiting"; exit 0; fi

NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" submit_e5_weak_monitor.sh)
echo "[$(date '+%F %T')] monitor $SLURM_JOB_ID started on $(hostname); successor $NEXT queued" >> "$R/monitor.log"

END=$((SECONDS + 3*3600 + 50*60))
while [ $SECONDS -lt $END ] && [ ! -f "$R/DONE" ]; do
  timeout 900 "$PY" -u incremental_experiments/e5_weak_monitor.py --once >> "$R/monitor_pass.log" 2>&1 \
    || echo "[$(date '+%F %T')] monitor pass exited with $?" >> "$R/monitor.log"
  sleep 300
done
