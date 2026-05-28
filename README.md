# Towards a Holistic Understanding of Selection Bias for Causal Effect Identification

Code for the paper [*Towards a Holistic Understanding of Selection Bias for Causal Effect Identification*](https://arxiv.org/abs/2605.13430).

The repository implements average treatment effect (ATE) estimators under both deterministic and undeterministic selection bias, and compares them against classical baselines (AIPW, TMLE, Heckman).

## Repository layout

```

├── sim_main.py              # Main simulation entry point
├── algorithm.py             # ATE estimation wrapper (AIPW, TMLE, Heckman, reweighted)
├── networks/                # Estimator implementations
│   ├── score_estimator.py       # Score-matching (naive + selection-corrected)
│   ├── gmm_estimator.py         # Gaussian mixture (naive + corrected)
│   ├── poly_estimator.py        # Polynomial regression
│   ├── IPW_estimator.py         # Trimmed inverse propensity weighting
│   ├── baseline_estimators.py
│   └── sel_funtion.py           # Selection probability estimation network
├── utils/
│   ├── data_generator.py    # Synthetic data with selection bias
│   ├── density_estimation.py
│   └── get_args.py          # Argument parsing
├── requirements.txt
├── run.sh                   # Sweep over noise distributions
```

## Installation

The code requires Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

or, with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -r requirements.txt
```

## Running

### Single run

```bash
python sim_main.py \
    --apply_sel_determin \
    --apply_sel_non_determin \
    --noise_type normal \
    --noise_func additive \
    --beta_center 1.5 \
    --beta_scale 3.0
```

### Sweep over noise distributions

`run.sh` runs the simulation for each of the four noise types reported in the paper (`normal`, `laplace`, `pareto`, `lognormal`):

```bash
bash run.sh
```


## Outputs

Each run produces, under `--log_dir`:

- `<run>_data.npz` — the generated dataset (observed + full population).
- `<run>_results.json` — ATE estimates from each method and the true ATE.
- `plot/<run>_plot.pdf` (or `.png`) — visualization of fitted potential outcomes.


## Citation

```bibtex
@article{causal_selection_bias,
  title  = {Towards a Holistic Understanding of Selection Bias for Causal Effect Identification},
  url    = {https://arxiv.org/abs/2605.13430}
}
```
