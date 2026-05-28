
import argparse
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description="ATE Estimation under Selection Bias")
    
    # Data Gen Args
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--noise_type', type=str, default='normal', choices=['normal', 'laplace', 'pareto', 'lognormal'], help='Noise distribution type')
    parser.add_argument('--noise_scale', type=float, default=0.5, help='Scale of the noise')
    parser.add_argument('--binary_outcome', action='store_true', default=False, help='Use binary outcome instead of continuous')
    
    # Selection Bias Flags
    parser.add_argument('--apply_sel_determin', action='store_true', default=False, help='Apply deterministic selection (remove tails)')
    parser.add_argument('--x_threshold', type=float, default=1.5, help='Threshold for deterministic selection (|x| > thresh)')
    
    parser.add_argument('--apply_sel_non_determin', action='store_true', default=False, help='Apply non-deterministic selection (probabilistic on Y)')

    parser.add_argument('--beta_center', type=float, default=1.5, help='Center for logistic selection function')
    parser.add_argument('--beta_scale', type=float, default=3.0, help='Scale/Steepness for logistic selection function')
    
    # Covariate dimensionality
    parser.add_argument('--x_dim', type=int, default=1, help='Dimensionality of covariate X')

    # Model Args
    parser.add_argument('--epochs', type=int, default=200, help='Epochs for Score Matching')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim for neural nets')
    parser.add_argument('--beta_reg', type=float, default=0.05, help='Regularization weight for selection correction')
    
    parser.add_argument('--log_dir', type=str, default="logs", help='Directory for logs')
    parser.add_argument('--noise_func', type=str, default="additive", choices=['additive', 'multiplicative'], help='Noise function type: additive or multiplicative')
    parser.add_argument('--x_func', type=str, default="polynomial", choices=['polynomial', 'sin', 'logfunc'], help='Functional form of X effect on Y')
    return parser.parse_args()

def generate_filename(args):
    # Construct a descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sel_str = f"Det{int(args.apply_sel_determin)}_NonDet{int(args.apply_sel_non_determin)}"
    fname = f"seed{args.seed}_N{args.num_samples}_Xdim{args.x_dim}_{args.noise_type}_scale{args.noise_scale}_{sel_str}_betaC{args.beta_center}_betaS{args.beta_scale}_{timestamp}"
    return fname

def get_args_gwas():
    parser = argparse.ArgumentParser(description="GWAS Simulation with Selection Bias")
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    # data
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples in the population')
    parser.add_argument('--g_dim', type=int, default=100, help='Dimensionality of genetic data')
    parser.add_argument('--y_dim', type=int, default=3, help='Dimensionality of outcomes')
    parser.add_argument('--noise_type', type=str, default='normal', choices=['normal', 'laplace', 'pareto', 'lognormal'], help='Noise distribution type')
    # selection bias
    parser.add_argument('--apply_sel_determin', action='store_true', default=True, help='Apply deterministic selection (remove tails)')
    parser.add_argument('--apply_sel_non_determin', action='store_true', default=False, help='Apply non-deterministic selection (probabilistic on Y)')
    # epochs and lr
    parser.add_argument('--epochs', type=int, default=400, help='Epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim for neural nets')
    parser.add_argument('--beta_reg', type=float, default=0.05, help='Regularization weight for selection correction')
    parser.add_argument('--log_dir', type=str, default="logsGWAS", help='Directory for logs')
    return parser.parse_args()

def generate_filename_gwas(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sel_str = f"Det{int(args.apply_sel_determin)}_NonDet{int(args.apply_sel_non_determin)}"
    fname = f"seed{args.seed}_N{args.num_samples}_G{args.g_dim}_Y{args.y_dim}_{args.noise_type}_{sel_str}_{timestamp}"
    return fname

def get_args_iv():
    parser = argparse.ArgumentParser(description="IV Estimation without Unobserved Confounders")
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    # data
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples in the population')
    parser.add_argument('--g_dim', type=int, default=50, help='Dimensionality of genetic/instrument data')
    parser.add_argument('--x_dim', type=int, default=3, help='Dimensionality of observed confounders')
    parser.add_argument('--noise_type', type=str, default='normal', choices=['normal', 'laplace', 'pareto', 'lognormal'], help='Noise distribution type')

    # selection bias
    parser.add_argument('--apply_sel_determin', action='store_true', default=True, help='Apply deterministic selection (remove tails)')
    parser.add_argument('--apply_sel_non_determin', action='store_true', default=False, help='Apply non-deterministic selection (probabilistic on Y)')

    # model
    parser.add_argument('--epochs', type=int, default=5000, help='Epochs for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim for neural nets')
    parser.add_argument('--beta_reg', type=float, default=0.01, help='Regularization weight for selection correction')
    parser.add_argument('--dsm_sigma', type=float, default=0.1, help='Perturbation sigma for DSM')
    parser.add_argument('--n_sample', type=int, default=10000, help='Number of samples to draw from fitted model for 2SLS')
    parser.add_argument('--log_dir', type=str, default="logsIV", help='Directory for logs')
    return parser.parse_args()

def generate_filename_iv(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sel_str = f"selDet_{int(args.apply_sel_determin)}_nonDet{int(args.apply_sel_non_determin)}"
    fname = f"seed{args.seed}_N{args.num_samples}_G{args.g_dim}_X{args.x_dim}_{args.noise_type}_{sel_str}_{timestamp}"
    return fname