from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks.score_estimator import ScoreEstimator
from tqdm import trange
import pdb
from networks.sel_funtion import SelectionFunction
# ================================
# ==========
# 1. The Conditional GMM (Mixture Density Network)
# ==========================================
class MDN(nn.Module):
    """
    Models P(Y|X) as a GMM where parameters depend on X.
    Output: pi(x), mu(x), sigma(x)
    """
    def __init__(self, input_dim=1, hidden_dim=64, n_components=5):
        super().__init__()
        self.n_components = n_components
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Heads for GMM parameters
        self.pi_head = nn.Linear(hidden_dim, n_components)
        self.mu_head = nn.Linear(hidden_dim, n_components)
        self.sigma_head = nn.Linear(hidden_dim, n_components)


    def forward(self, x):
        # x : [batch_size, input_dim]
        # breakpoint()
        
        h = self.feature_extractor(x)
        
        # Mixing coefficients: Softmax to sum to 1
        pi = F.softmax(self.pi_head(h), dim=1)
        
        # Means: No constraints (linear)
        mu = self.mu_head(h)
        
        # Sigmas: Positive (Exp)
        # Add epsilon for numerical stability
        sigma = torch.exp(self.sigma_head(h)) + 1e-6
        
        return pi, mu, sigma

    def loss(self, x, y, weights=None):
        pi, mu, sigma = self.forward(x)
        y_exp = y.expand_as(mu)
        m = torch.distributions.Normal(mu, sigma)
        log_prob = m.log_prob(y_exp)
        
        # LogSumExp
        weighted_log_prob = torch.log(pi + 1e-8) + log_prob
        log_likelihood = torch.logsumexp(weighted_log_prob, dim=1)
        
        if weights is not None:
            return -torch.mean(weights * log_likelihood)
        
        return -torch.mean(log_likelihood)
    
# ==========================================
# 2. The Combined Estimator
# ==========================================
class SeparateGMMEstimator:
    """ 
    Separate Estimator: P(X) via GMM, P(Y|X) via MDN 
    Controlled by flag: use_correction
    """
    def __init__(self, n_components=5, seed=42, use_correction=False, 
                 hidden_dim=64, lr=0.005, epochs=300, beta_reg=0.01):
        self.n_comp = n_components
        self.seed = seed
        self.use_correction = use_correction # <--- The FLAG for correction beta
        self.hidden_dim, self.lr, self.epochs, self.beta_reg = hidden_dim, lr, epochs, beta_reg
        
        self.gmm_x = GaussianMixture(n_components=n_components, random_state=seed)
        self.mdn_yx = MDN(hidden_dim=self.hidden_dim, n_components=self.n_comp)
        self.x_support_min = -3.0
        self.x_support_max = 3.0
        self.selection_model = SelectionFunction()

        # self.mdn_yx = None

    def fit(self, X, Y, T):
        # weights = None
        
        # # 1. Apply Correction (If Flag is True)
        # if self.use_correction:
        #     print("   > [SeparateGMM] Learning correction weights...")
        #     helper = ScoreEstimator(epochs=self.epochs, correct_selection=True, hidden_dim=self.hidden_dim, 
        #                             lr=self.lr, beta_reg=self.beta_reg)
        #     weights = helper.get_weights(X, Y, T)

        # # 2. Fit Marginal P(X)
        # if self.use_correction and weights is not None:
        #     # Resample X based on weights
        #     p = weights / weights.sum()
        #     idx = np.random.choice(len(X), size=len(X)*5, p=p)
        #     self.gmm_x.fit(X[idx])
        # else:
        #     # Standard fit
        #     self.gmm_x.fit(X)
        
        self.gmm_x.fit(X)
        # 3. Fit Conditional P(Y|X)
        if self.use_correction:
            opt = optim.Adam(list(self.mdn_yx.parameters()) + list(self.selection_model.parameters()), lr=self.lr)
        else:
            opt = optim.Adam(self.mdn_yx.parameters(), lr=self.lr)
        
        X_t = torch.FloatTensor(X)
        Y_t = torch.FloatTensor(Y).reshape(-1, 1)
        T_t = torch.FloatTensor(T).reshape(-1, 1)
        # W_t = torch.FloatTensor(weights).reshape(-1, 1) if weights is not None else None
        
        # Bundle data
        tensors = [X_t, Y_t]
        # if W_t is not None: tensors.append(W_t)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=64, shuffle=True)
        
        for epoch in trange(self.epochs):
            for batch in loader:
                bx, by = batch[0], batch[1]
                # If using correction, batch[2] contains weights
                # bw = batch[2] if self.use_correction else None
                weights = None
                if self.use_correction:
                    beta = self.selection_model(X_t, Y_t, T_t)
                            
                    # Safe Division
                    weights = 1.0 / (beta + 1e-6)


                opt.zero_grad()
                loss = self.mdn_yx.loss(bx, by, weights=weights)
                loss.backward()
                opt.step()
                if epoch % 50 == 0:
                    # pass
                    print(f"Loss: {loss.item():.4f}")

    # def predict_mean(self, X_eval):
    #     self.mdn_yx.eval()
    #     with torch.no_grad():
    #         pi, mu, sigma = self.mdn_yx(torch.FloatTensor(X_eval))
    #         # E[Y|X] = Sum(pi * mu)
    #         return torch.sum(pi * mu, dim=1).numpy()

    def predict_mean(self, n_samples=1000, X_samples=None):
        """
        Calculates E[Y|X] = Sum_k [ pi_k(x) * mu_k(x) ]
        """
        
        # 1. Sample X from Marginal GMM
        if X_samples is None:
            X_samples, _ = self.gmm_x.sample(n_samples)
        
        # keep the one between X_min and X_max
        X_min, X_max = self.x_support_min, self.x_support_max
        X_samples = X_samples[(X_samples >= X_min) & (X_samples <= X_max)].reshape(-1, 1)
        # print a numpy histogram of X_samples
        hist, bin_edges = np.histogram(X_samples, bins=30)
        print("Histogram of sampled X from GMM:", hist)

        X_ts = torch.FloatTensor(X_samples)
        
        with torch.no_grad():
            pi, mu, sigma = self.mdn_yx(X_ts)
            # Expectation of a mixture is sum(weight * component_mean)
            expected_y = torch.sum(pi * mu, dim=1)
            
        return expected_y.numpy()

    def sample_from_joint(self, n_samples=1000):
        """
        Generative Process:
        1. Sample x ~ P(X)
        2. Sample y ~ P(Y|x)
        """
        # 1. Sample X from Marginal GMM
        X_samples, _ = self.gmm_x.sample(n_samples)
        X_ts = torch.FloatTensor(X_samples)
        
        # 2. Sample Y from Conditional MDN
        with torch.no_grad():
            pi, mu, sigma = self.mdn_yx(X_ts)
            
            # Select component for each sample
            # categorical sampling based on pi
            categorical = torch.distributions.Categorical(pi)
            comp_indices = categorical.sample() # [n_samples]
            
            # Gather specific mu and sigma for the chosen component
            # fancy indexing
            rows = torch.arange(n_samples)
            mu_sel = mu[rows, comp_indices]
            sigma_sel = sigma[rows, comp_indices]
            
            # Sample gaussian
            Y_samples = torch.normal(mu_sel, sigma_sel).numpy().reshape(-1, 1)
            
        return X_samples, Y_samples