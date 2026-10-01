#!/bin/bash
cd /net/projects/ranalab/rajhansini/replication_16features
export OMP_NUM_THREADS=1
/usr/bin/time -v /net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python -u incremental_experiments/e5_weak.py \
  --stage eval --genes 3 --unit "$1" --limit 2 --out incremental_experiments/results/e5_weak_evaltest 2>&1 \
  | grep -v "scalar ·\|materialization\|joint tuples\|Warning\|from .Struct"
