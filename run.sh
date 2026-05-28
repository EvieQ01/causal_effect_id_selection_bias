#!/usr/bin/env bash
# Runs the simulation across the four noise distributions reported in the paper.
for noise_type in "polynomial" "laplace" "pareto" "lognormal"; do
    python sim_main.py \
        --apply_sel_determin \
        --apply_sel_non_determin \
        --noise_type "$noise_type" \
        --noise_func additive \
        --beta_center 2.0 \
        --beta_scale 3.0
done
