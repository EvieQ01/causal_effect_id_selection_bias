import numpy as np
# --- Standard Estimators ---
class TrimmedIPWEstimator:
    def fit(self, X, Y, T, weights=None):
        if weights is None: weights = np.ones_like(Y)
        self.mean = np.sum(Y * weights) / (np.sum(weights) + 1e-8)
    def predict_mean(self, X_eval):
        return np.full(X_eval.shape[0], self.mean)
    def predict_beta(self, X=None, Y=None, T=None):
        return np.ones(len(X))
