import numpy as np
import torch
from networks.score_estimator import ScoreEstimator
from networks.gmm_estimator import GMMEstimator
from networks.gmm_estimator_sep import SeparateGMMEstimator
from networks.poly_estimator import PolyEstimator
from networks.IPW_estimator import TrimmedIPWEstimator
from algorithm import Algorithm_ate_sel
import matplotlib.pyplot as plt
from utils.data_generator import generate_selection_bias_data, get_true_means
from utils.get_args import get_args, generate_filename
import os
import json
import matplotlib.font_manager as fm




# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    args = get_args()
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # Update this path as needed
    # if i'm using Mac
    if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
        font_path = '/Library/Fonts/Times New Roman.ttf'
    times_font = fm.FontProperties(fname=font_path)
    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Logs
    log_dir = args.log_dir
    plot_dir = os.path.join(log_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    fbase = generate_filename(args)
    
    # Data
    print(f"Generating data ({args.noise_type})...")
    data = generate_selection_bias_data(n=args.num_samples,
                                        x_dim=args.x_dim,
                                        x_threshold=args.x_threshold,
                                        beta_scale=args.beta_scale,
                                        beta_center=args.beta_center,
                                        noise_type=args.noise_type,
                                        binary_outcome=args.binary_outcome,
                                        apply_sel_determin=args.apply_sel_determin,
                                        apply_sel_non_determin=args.apply_sel_non_determin,
                                        noise_scale=args.noise_scale,
                                        noise_func=args.noise_func,
                                        x_func=args.x_func)
    np.savez(os.path.join(log_dir, f"{fbase}_data.npz"), **data)
    
    # Config Estimators with explicit args
    estimators = [
        ("Trimmed IPW", TrimmedIPWEstimator, {}),
        ("Polynomial", PolyEstimator, {}),
        ("Naive GMM", GMMEstimator, {'n_components':5, 'seed':args.seed, 'use_correction':False, 'x_dim':args.x_dim}),
        ("Corrected GMM", GMMEstimator, {'n_components':5, 'seed':args.seed, 'use_correction':True,
                                         'hidden_dim':args.hidden_dim, 'lr':args.lr, 'epochs':args.epochs, 'beta_reg':args.beta_reg, 'x_dim':args.x_dim}),
        ("Naive SM", ScoreEstimator, {'hidden_dim':args.hidden_dim, 'lr':args.lr, 'epochs':args.epochs, 'correct_selection':False, 'x_dim':args.x_dim}),
        ("Corrected SM", ScoreEstimator, {'hidden_dim':args.hidden_dim, 'lr':args.lr, 'epochs':args.epochs, 'correct_selection':True, 'beta_reg':args.beta_reg, 'x_dim':args.x_dim}),
    ]
    
    algo = Algorithm_ate_sel(args.seed)
    res = {'args': vars(args), 'true_ate': float(data['true_ate']), 'estimates': {}}
    
    print(f"True ATE: {data['true_ate']:.4f}")
    
    # ── Baseline estimators ───────────────────────────────────────────────────
    for bname, bmethod in [("AIPW Oracle", lambda d: algo.run_aipw(d, oracle=True)),
                           ("AIPW", algo.run_aipw), ("TMLE", algo.run_tmle), ("Heckman", algo.run_heckman)]:
        print(f"Running {bname}...")
        try:
            ate = bmethod(data)
            res['estimates'][bname] = float(ate)
            print(f"  > {ate:.4f} (Err: {abs(ate - data['true_ate']):.4f})")
        except Exception as e:
            print(f"  > Failed: {e}")
            res['estimates'][bname] = "Failed"

    # Visualization Grid (1D only)
    do_viz = (args.x_dim == 1)
    X_viz = np.linspace(-3.5, 3.5, 200).reshape(-1, 1) if do_viz else None
    tm0, tm1 = (get_true_means(X=X_viz) if do_viz else (None, None))

    # 1. Run Dummy to get Propensity / Overlap info
    print("Running dummy pass for Propensity/Overlap info...")
    _, _, _, marg_est, clf_prop = algo.run(data, PolyEstimator, X_viz=None)

    # Calculate Overlap (1D only)
    overlap_mask = None
    if do_viz:
        e_viz = clf_prop.predict_proba(X_viz)[:, 1]
        overlap_mask = (e_viz >= 0.05) & (e_viz <= 0.95)

    if do_viz:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams['mathtext.fontset'] = 'stix'
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes_flat = axes.flatten()

    for i, (name, Cls, kwargs) in enumerate(estimators):
        print(f"Running {name}...")
        try:
            ate, mu1_viz, mu0_viz, _, _ = algo.run(data, Cls, X_viz=X_viz, **kwargs)
            res['estimates'][name] = float(ate)
            print(f"  > {ate:.4f} (Err: {abs(ate - data['true_ate']):.4f})")

            if do_viz:
                ax = axes_flat[i]
                t0_mask = data['T'] == 0
                t1_mask = data['T'] == 1
                ax.scatter(data['X'][t0_mask], data['Y'][t0_mask], c='b', alpha=0.1, s=5, label='Obs T=0' if i==0 else None)
                ax.scatter(data['X'][t1_mask], data['Y'][t1_mask], c='orange', alpha=0.1, s=5, label='Obs T=1' if i==0 else None)
                ax.plot(X_viz, tm0, 'b--', label='True Y(0)' if i==0 else None)
                ax.plot(X_viz, tm1, color='orange', linestyle='--', label='True Y(1)' if i==0 else None)
                if mu0_viz is not None:
                    ax.plot(X_viz, mu0_viz, 'b-', alpha=0.9, label='Est Y(0)' if i==0 else None)
                    ax.plot(X_viz, mu1_viz, 'orange', linestyle='-', alpha=0.9, label='Est Y(1)' if i==0 else None)
                if overlap_mask is not None and np.any(overlap_mask):
                    x_flat = X_viz.flatten()
                    in_region = False; start = None
                    for k, val in enumerate(overlap_mask):
                        if val and not in_region:
                            in_region = True; start = x_flat[k]
                        elif not val and in_region:
                            in_region = False
                            ax.axvspan(start, x_flat[k], color='green', alpha=0.1, label='Overlap' if (i==0 and k==0) else None)
                    if in_region:
                        ax.axvspan(start, x_flat[-1], color='green', alpha=0.1)
                ax.set_title(f"{name}\nATE: {ate:.2f} (True: {data['true_ate']:.2f})", fontsize=30)
                ax.title.set_fontproperties(times_font)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend()
                if times_font is not None:
                    for label in ax.get_xticklabels():
                        label.set_fontproperties(times_font); label.set_fontsize(20)
                    for label in ax.get_yticklabels():
                        label.set_fontproperties(times_font); label.set_fontsize(20)

        except Exception as e:
            print(f"  > Failed: {e}")
            res['estimates'][name] = "Failed"
            if do_viz:
                axes_flat[i].text(0.5, 0.5, f"Failed: {e}", ha='center')

    with open(os.path.join(log_dir, f"{fbase}_results.json"), 'w') as f:
        json.dump(res, f, indent=4)

    if do_viz:
        plt.tight_layout()
        print(f"Saving log plot as {os.path.join(plot_dir, f'{fbase}_plot.pdf')}...")
        if log_dir.endswith("single"):
            plt.savefig(os.path.join(plot_dir, f"{fbase}_plot.pdf"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(plot_dir, f"{fbase}_plot.png"))
    print("Done.")