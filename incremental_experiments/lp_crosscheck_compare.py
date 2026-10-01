"""Compare Kanix's LP exact DP (submit_lp_crosscheck.sh) with the sweep's backward induction.

Each lp_crosscheck/rowNNNN.json is the output of his unmodified script with PuLP
available; rowNNNN is the myopic_weakness_sweep row with the same config.
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
LP, SWEEP = RES / "lp_crosscheck", RES / "myopic_weakness_sweep"

print(f"{'row':<9}{'V* LP':>13}{'V* primal':>13}{'|diff|':>9}{'stop |diff|':>12}{'myopic |diff|':>14}"
      f"  {'DP root LP / primal':<34}{'myopic root':<12}{'gurobi log':<10}")
for f in sorted(LP.glob("row*.json")):
    lp = json.loads(f.read_text())[0]
    sw = json.loads(next(SWEEP.glob(f"{f.stem[3:]}_*.json")).read_text())
    log = LP / f"{f.stem}_gurobi.log"
    ran = log.exists() and "Optimal objective" in log.read_text()
    same_dp = lp["exact_root_action"][:2] == sw["root_action_dp"]
    same_myo = lp["myopic_root_action"][:2] == sw["root_action_myopic"]
    print(f"{f.stem:<9}{lp['exact_root_value']:>13.9f}{sw['V_root']:>13.9f}"
          f"{abs(lp['exact_root_value'] - sw['V_root']):>9.1e}"
          f"{abs(lp['stop_value'] - sw['V_stop_root']):>12.1e}"
          f"{abs(lp['myopic_root_value'] - sw['L_myopic']):>14.1e}"
          f"  {str(lp['exact_root_action'][:2]) + (' ==' if same_dp else ' !=') :<34}"
          f"{'same' if same_myo else 'DIFF':<12}{'LP ran' if ran else 'NO LOG':<10}")
