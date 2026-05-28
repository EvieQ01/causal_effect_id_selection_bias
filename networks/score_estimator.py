
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from networks.sel_funtion import SelectionFunction
class ScoreNet(nn.Module):
    def __init__(self, x_dim=1, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, n_hidden),  # x_dim + y(1)
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def corrected_score_matching_loss(score_model, selection_model, x, y, t, beta_reg=0.1, l2_reg=0.01):
    y = y.clone().detach().requires_grad_(True)
    
    # 1. Score of underlying density
    score_true = score_model(x, y)
    
    # 2. Score of selection prob (beta)
    beta_val = selection_model(x, y, t)
    
    # Add epsilon for log stability
    log_beta = torch.log(beta_val + 1e-6)
    d_log_beta_dy = torch.autograd.grad(log_beta, y, torch.ones_like(log_beta), create_graph=True)[0]
    
    # 3. Composite Score
    score_obs = score_true + d_log_beta_dy
    
    # 4. Hyvarinen Loss
    d_score_obs_dy = torch.autograd.grad(score_obs, y, torch.ones_like(score_obs), create_graph=True)[0]
    sm_loss = 0.5 * (score_obs ** 2) + d_score_obs_dy
    
    # 5. Maximizing Likelihood of Selection
    likelihood_term = -log_beta
    
    # 6. Regularization on Beta magnitude (Prevent explosion)
    beta_magnitude_penalty = torch.mean(beta_val ** 2)
    
    return torch.mean(sm_loss + beta_reg * likelihood_term) + l2_reg * beta_magnitude_penalty


class ScoreEstimator:
    def __init__(self, epochs=300,
                 lr=0.005,
                 hidden_dim=64, correct_selection=True, beta_reg=0.1, x_dim=1):
        self.epochs = epochs
        self.lr = lr
        self.correct_selection = correct_selection
        self.score_model = None
        self.selection_model = None
        self.y_range = None
        self.hidden_dim = hidden_dim
        self.beta_reg = beta_reg
        self.x_dim = x_dim
        self.score_model = ScoreNet(x_dim=x_dim, n_hidden=self.hidden_dim)
        self.selection_model = SelectionFunction(x_dim=x_dim)


    def fit(self, X, Y, T):
        self.y_range = (Y.min(), Y.max())
        X_ts = torch.FloatTensor(X)
        Y_ts = torch.FloatTensor(Y).reshape(-1, 1)
        T_ts = torch.FloatTensor(T).reshape(-1, 1)
        
        dataset = torch.utils.data.TensorDataset(X_ts, Y_ts, T_ts)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        
        params = list(self.score_model.parameters()) + list(self.selection_model.parameters())
        optimizer = optim.Adam(params, lr=self.lr)
        
        for _ in trange(self.epochs):
            for bx, by, bt in loader:
                optimizer.zero_grad()
                if self.correct_selection:
                    # Added l2_reg parameter here
                    # breakpoint()
                    loss = corrected_score_matching_loss(self.score_model, self.selection_model, bx, by, bt, l2_reg=self.beta_reg)
                else:
                    by.requires_grad_(True)
                    score = self.score_model(bx, by)
                    d_score_dy = torch.autograd.grad(score.sum(), by, create_graph=True)[0]
                    loss = torch.mean(0.5 * score**2 + d_score_dy)
                
                loss.backward()
                # Gradient Clipping to prevent NaN
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

    def predict_mean(self, X_eval):
        margin = 3.0
        y_grid = torch.linspace(self.y_range[0]-margin, self.y_range[1]+margin, 200).unsqueeze(1)
        dy = (y_grid[1] - y_grid[0]).item()
        X_eval_ts = torch.FloatTensor(X_eval)
        means = []
        with torch.no_grad():
            for i in range(len(X_eval_ts)):
                x_val = X_eval_ts[i].unsqueeze(0).repeat(len(y_grid), 1)
                scores = self.score_model(x_val, y_grid)
                log_p = torch.cumsum(scores, dim=0) * dy
                p_unnorm = torch.exp(log_p - log_p.max())
                p_norm = p_unnorm / (torch.sum(p_unnorm) * dy + 1e-8)
                means.append(torch.sum(y_grid * p_norm).item() * dy)
        return np.array(means)

    def predict_beta(self, X=None, Y=None, T=None):
        if not self.correct_selection:
            return np.ones(len(X))
        
        self.selection_model.eval()
        with torch.no_grad():
            tx = torch.FloatTensor(X)
            ty = torch.FloatTensor(Y).view(-1, 1)
            
            if isinstance(T, float) or isinstance(T, int):
                tt = torch.full((len(X), 1), float(T))
            else:
                 # Check if T is a scalar-like array/tensor
                if hasattr(T, 'shape') and (len(T.shape) == 0 or (len(T.shape)==1 and T.shape[0]==1)):
                     tt = torch.full((len(X), 1), float(T))
                else:
                     # Assume T matches X length
                     tt = torch.FloatTensor(T).view(-1, 1)

            beta = self.selection_model(tx, ty, tt)
            beta_np = beta.numpy().flatten()
            return beta_np

    def get_weights(self, X, Y, T):
        # Compute selection probabilities beta(X,Y,T) as weights
        self.selection_model.eval()
        with torch.no_grad():
            X_ts = torch.FloatTensor(X)
            Y_ts = torch.FloatTensor(Y).reshape(-1, 1)
            T_ts = torch.FloatTensor(T).reshape(-1, 1)
            
            beta = self.selection_model(X_ts, Y_ts, T_ts)
            beta_np = beta.numpy().flatten()
            
            # Safe Division
            weights = 1.0 / (beta_np + 1e-6)
            
            # Check for NaNs or Infs
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                print("Warning: NaNs/Infs detected in weights. Replacing with mean.")
                weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
            
            # Normalize
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            return weights / weights.sum()
