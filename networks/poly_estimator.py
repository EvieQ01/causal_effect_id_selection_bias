from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

class PolyEstimator:
    def __init__(self, degree=3):
        self.poly = PolynomialFeatures(degree); self.reg = LinearRegression()
    
    def fit(self, X, Y, T):
        self.reg.fit(self.poly.fit_transform(X), Y)
    
    def predict_mean(self, X_eval):
        return self.reg.predict(self.poly.transform(X_eval))

    def predict_beta(self, X=None, Y=None, T=None):
        return np.ones(len(X))
