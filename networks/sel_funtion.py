import torch
import torch.nn as nn

class SelectionFunction(nn.Module):
    def __init__(self, x_dim=1, n_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(x_dim + 2, n_hidden),  # x_dim + y(1) + t(1)
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x, y, t):
        out = self.mlp(torch.cat([x, y, t], dim=1))
        # Stabilize exp: clamp the input to avoid overflow/underflow
        out = torch.clamp(out, min=-10.0, max=10.0) 
        return torch.exp(out)

class SelectionNetworkGWAS(nn.Module):
    """
    Neural network to estimate the selection correction factor beta(G, Y).
    """
    def __init__(self, g_dim=100, y_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim + y_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, g, y):
        cat_in = torch.cat([g, y], dim=1)
        out = self.net(cat_in)
        # Clamp for numerical stability before exponentiation
        clamped = torch.clamp(out, -10, 10)
        return torch.exp(clamped)
