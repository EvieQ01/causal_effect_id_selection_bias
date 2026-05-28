
from sklearn.mixture import GaussianMixture
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import torch

class MarginalDensityEstimator:
    def __init__(self):
        self.model = GaussianMixture(n_components=10, random_state=42)

    def fit(self, X=None):
        self.model.fit(X)

    def predict_density(self, X_eval=None):
        scores = self.model.score_samples(X_eval)
        return np.exp(scores)

    def sample(self, n_samples=1000):
        sample_res = self.model.sample(n_samples)
        X_gen = sample_res[0]
        return X_gen.astype(np.float32)

class IndependentCategoricalEstimator(nn.Module):
    """
    Estimates the marginal density P(G) assuming independence across the g_dim dimensions.
    G is assumed to be categorical, taking integer values in [0, num_classes-1].
    """
    def __init__(self, g_dim=100, num_classes=3):
        super().__init__()
        self.g_dim = g_dim
        self.num_classes = num_classes
        
        # We store the distribution parameters as logits (unnormalized log probabilities)
        # Shape: (g_dim, num_classes)
        self.logits = nn.Parameter(torch.zeros(g_dim, num_classes))

    def fit(self, G_obs):
        """
        Fits the marginal distribution using empirical frequencies (MLE).
        
        Args:
            G_obs (torch.Tensor): Observed G data of shape (n_samples, g_dim).
                                  Must contain values 0, 1, ..., num_classes-1.
        """
        # Ensure G_obs is treated as integer indices for counting
        G_idx = G_obs.long()
        counts = torch.zeros(self.g_dim, self.num_classes, device=G_obs.device)
        
        for c in range(self.num_classes):
            # Count occurrences of class `c` across all samples, for each dimension
            counts[:, c] = (G_idx == c).sum(dim=0)
            
        # Apply Laplace (add-one) smoothing to prevent log(0) for unobserved categories
        smoothed_counts = counts + 1.0 
        probs = smoothed_counts / smoothed_counts.sum(dim=1, keepdim=True)
        
        # Update the logits parameter in-place
        with torch.no_grad():
            self.logits.copy_(torch.log(probs))
            
    def log_prob(self, G):
        """
        Computes the log probability mass function \log P(G) for a batch of G.
        Because dimensions are independent: \log P(G) = \sum_{i=1}^{g_dim} \log P(G_i)
        
        Args:
            G (torch.Tensor): Tensor of shape (n_samples, g_dim)
            
        Returns:
            torch.Tensor: Log probabilities of shape (n_samples,)
        """
        dist = Categorical(logits=self.logits)
        # dist.log_prob evaluates the log prob for each dimension independently.
        # We sum across the g_dim axis (dim=1) to get the joint log probability.
        return dist.log_prob(G.long()).sum(dim=1)

    def sample(self, n_samples):
        """
        Samples a new batch of G from the estimated marginal distribution.
        
        Args:
            n_samples (int): Number of samples to generate.
            
        Returns:
            torch.Tensor: Sampled G of shape (n_samples, g_dim) with float dtype.
        """
        dist = Categorical(logits=self.logits)
        # dist.sample() returns shape (n_samples, g_dim)
        return dist.sample((n_samples,)).float()