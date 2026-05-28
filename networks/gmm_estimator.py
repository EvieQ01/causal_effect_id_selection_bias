import numpy as np
from sklearn.mixture import GaussianMixture
from networks.score_estimator import ScoreEstimator
import torch
import torch.nn as nn
from torch.distributions import Normal
class GMMEstimator:
    def __init__(self, n_components=5,
                 use_correction=False,
                 seed=42,
                 hidden_dim=64,
                 lr=0.005,
                 epochs=300,
                 beta_reg=0.1,
                 x_dim=1):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=seed)
        self.use_correction = use_correction
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.beta_reg = beta_reg
        self.x_dim = x_dim
        self.x_support_min = -3.0
        self.x_support_max = 3.0
        self.helper = None

    def fit(self, X, Y, T):
        data = np.hstack([X, Y.reshape(-1, 1)])
        # pdb.set_trace()
        if self.use_correction:
            print("   > Learning weights for GMM correction...")
            self.helper = ScoreEstimator(epochs=self.epochs, correct_selection=True, hidden_dim=self.hidden_dim,
                                    lr=self.lr, beta_reg=self.beta_reg, x_dim=self.x_dim)
            self.helper.fit(X, Y, T)
            weights = self.helper.get_weights(X, Y, T)
            
            # Safe Resampling
            try:
                indices = np.random.choice(len(X), size=len(X)*5, p=weights)
                data = data[indices]
            except ValueError as e:
                print(f"Resampling failed: {e}. Falling back to unweighted.")
                
        self.gmm.fit(data)

    def predict_beta(self, X=None, Y=None, T=None):
        if self.use_correction and self.helper is not None:
             return self.helper.predict_beta(X=X, Y=Y, T=T)
        return np.ones(len(X))

    def predict_mean(self, X_eval):
        means = []
        W, M, C = self.gmm.weights_, self.gmm.means_, self.gmm.covariances_
        xd = self.x_dim
        n_comp = len(W)
        for x in X_eval:
            x = np.array(x).flatten()[:xd]   # shape (xd,)
            p_xk = np.zeros(n_comp)
            cond_mu_k = np.zeros(n_comp)
            for k in range(n_comp):
                mu_x  = M[k, :xd]              # (xd,)
                mu_y  = M[k, xd]               # scalar
                S_xx  = C[k, :xd, :xd]         # (xd, xd)
                S_yx  = C[k, xd, :xd]          # (xd,)
                diff  = x - mu_x
                sign, logdet = np.linalg.slogdet(S_xx)
                inv_S_xx = np.linalg.inv(S_xx)
                mahal = diff @ inv_S_xx @ diff
                p_xk[k] = np.exp(-0.5 * mahal - 0.5 * logdet
                                 - 0.5 * xd * np.log(2 * np.pi))
                cond_mu_k[k] = mu_y + S_yx @ inv_S_xx @ diff
            numer = np.sum(W * p_xk * cond_mu_k)
            denom = np.sum(W * p_xk)
            means.append(numer / (denom + 1e-8) if denom > 0 else 0)
        return np.array(means)

    def get_marginal_expectation(self):
        """
        Analytically calculates E[Y] from the fitted GMM parameters.
        Since this estimator is trained on T=t, this returns E[Y(t)].
        """
        # 1. Extract Weights (pi_k)
        weights = self.gmm.weights_
        
        # 2. Extract Y-means (The 2nd dimension, index 1)
        # means shape is (n_components, 2) -> (x_mean, y_mean)
        # pdb.set_trace()
        means_y = self.gmm.means_[:, 1]
        
        # 3. Weighted Sum
        ey = np.sum(weights * means_y)
        return ey

    def sample_y(self, n_samples=3000):
        # Sample joint (X,Y) and return Y
        X_j, Y_j = self.gmm.sample(n_samples)[0][:, 0:1], self.gmm.sample(n_samples)[0][:, 1:2]
        return Y_j
    
    # --- MONTE CARLO VERSION (using your X sampling) ---
    def get_marginal_expectation_mc(self, n_samples=5000):
        """
        Calculates E[Y] by sampling X from the GMM marginal 
        and averaging the conditional expectations.
        """
        # 1. Sample X from the marginal P(X) of the GMM
        weights = self.gmm.weights_
        means_x = self.gmm.means_[:, 0]
        stds_x = np.sqrt(self.gmm.covariances_[:, 0, 0])
        
        indices = np.random.choice(len(weights), size=n_samples, p=weights)
        X_samples = np.random.normal(loc=means_x[indices], scale=stds_x[indices]).reshape(-1, 1)
        
        # only keep samples within support
        X_samples = X_samples[(X_samples >= self.x_support_min) & (X_samples <= self.x_support_max)]
        # 2. Predict E[Y|X] for these samples
        y_preds = self.predict_mean(X_samples)
        
        # 3. Average
        return np.mean(y_preds)

# Torch version of Mixture of Gaussians 
class MixtureOfGaussians(nn.Module):
    """
    Learnable diagonal Mixture of Gaussians for modeling residual/noise distribution.
    Models p(eps) where eps in R^d with K components.
    """
    def __init__(self, n_components=5, y_dim=1, trainable_means=False):
        super().__init__()
        self.n_components = n_components
        self.log_weights = nn.Parameter(torch.zeros(n_components))
        if trainable_means:
            self.means = nn.Parameter(torch.zeros(n_components, y_dim))
        else:
            # Fixed at zero: suitable for zero-mean noise residuals.
            self.register_buffer('means', torch.zeros(n_components, y_dim))
        self.log_scales = nn.Parameter(torch.zeros(n_components, y_dim))

    def log_prob(self, x):
        # x: (n, d)
        # log_weights: (K,), means: (K, d), log_scales: (K, d)
        log_pi = self.log_weights - torch.logsumexp(self.log_weights, dim=0)  # (K,)
        scales = torch.exp(self.log_scales).clamp(min=1e-4)  # (K, d)

        # x: (n, 1, d), means/scales: (1, K, d)
        x_exp = x.unsqueeze(1)
        means_exp = self.means.unsqueeze(0)
        scales_exp = scales.unsqueeze(0)

        # log N(x; mu_k, sigma_k) summed over d: (n, K)
        log_p_components = Normal(means_exp, scales_exp).log_prob(x_exp).sum(dim=-1)

        # log sum_k pi_k * p_k(x): (n,)
        return torch.logsumexp(log_pi + log_p_components, dim=1)

    def sample(self, n_samples):
        log_pi = self.log_weights - torch.logsumexp(self.log_weights, dim=0)
        k = torch.multinomial(log_pi.exp(), n_samples, replacement=True)  # (n,)
        means_k = self.means[k]                                            # (n, d)
        scales_k = torch.exp(self.log_scales[k]).clamp(min=1e-4)          # (n, d)
        return means_k + scales_k * torch.randn_like(scales_k)

