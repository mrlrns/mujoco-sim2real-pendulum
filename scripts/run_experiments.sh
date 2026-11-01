#!/usr/bin/env bash
# Reproduce every model, sweep and figure of the project.
# 110 policies x ~30 s (500k PPO steps on a laptop CPU), then the sweeps -> roughly one to two hours.
# Already-trained models are skipped.
set -e
mkdir -p models
train() {  # train <out> <args...>
  out=$1; shift
  [ -f "models/$out.zip" ] || python -m scripts.train "$@" --out "models/$out"
}

for s in 0 1 2 3 4 5 6 7 8 9; do
  # standard observation [cos, sin, omega]
  train base_s$s                       --family base --seed $s
  train base300k_s$s                   --family base --seed $s --steps 300000
  train dr_s$s                         --family dr   --seed $s
  train lat2_s$s                       --family lat  --lat-max 2 --seed $s
  train lat5_s$s                       --family lat  --lat-max 5 --seed $s
  train full_s$s                       --family full --seed $s
  # observation with the previous action
  train apbase_s$s    --prev           --family base --seed $s
  train apdr_s$s      --prev           --family dr   --seed $s
  train aplat3_s$s    --prev           --family lat  --lat-max 3 --seed $s
  train aplat5_s$s    --prev           --family lat  --lat-max 5 --seed $s
  train apfull_s$s    --prev           --family full --seed $s
done

python -m scripts.sweep_gear --mods models/base_s*.zip models/base300k_s*.zip models/dr_s*.zip --tag all
python -m scripts.sweep_latency --families base,dr,lat2,lat5,full --out results/data/latency_standard.csv
python -m scripts.sweep_latency --prefix ap --families base,dr,lat3,lat5,full --out results/data/latency_prev_action.csv

python -m scripts.plot_gear results/data/gear_sweep_all.csv
python -m scripts.plot_compare
