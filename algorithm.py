import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import norm
from networks.marginal_estimator import MarginalDensityEstimator


# ==========================================
# Algorithm Wrapper
# ==========================================
class Algorithm_ate_sel:
    def __init__(self, seed=42):
        self.seed = seed

    def run(self, data, Estimator, X_viz=None, **kwargs):
        X = data['X']
        Y = data['Y']
        T = data['T']

        # 1. Marginal P(X)
        marginal_est = MarginalDensityEstimator()
        marginal_est.fit(X=X)
        X_gen = marginal_est.sample(n_samples=5000)

        # 2. Propensity
        nn_clf = MLPClassifier((32,32), max_iter=500, random_state=self.seed)
        clf = CalibratedClassifierCV(nn_clf)
        clf.fit(X, T)
        e = clf.predict_proba(X)[:, 1]

        # Define overlap region
        c = 0.05
        mask1 = (T==1) & (e >= c)
        mask0 = (T==0) & (e <= 1-c)

        m1 = Estimator(**kwargs)
        m0 = Estimator(**kwargs)

        # 3. Fit
        if Estimator.__name__ == "TrimmedIPWEstimator":
            mask1 = (e <= 1-c) & (e >= c) & (T==1)
            mask0 = (e <= 1-c) & (e >= c) & (T==0)

            denom1 = e[mask1] + 1e-6
            w1 = 1 / denom1

            term0 = 1 - e[mask0]
            denom0 = term0 + 1e-6
            w0 = 1 / denom0

            m1.fit(X=X[mask1], Y=Y[mask1], T=T[mask1], weights=w1)
            m0.fit(X=X[mask0], Y=Y[mask0], T=T[mask0], weights=w0)

            # Assuming TrimmedIPWEstimator has 'mean' attribute as per sim_main
            ate = m1.mean - m0.mean

            mu1_viz = None
            mu0_viz = None
            if X_viz is not None:
                mu1_viz = m1.predict_mean(X_eval=X_viz)
                mu0_viz = m0.predict_mean(X_eval=X_viz)

            return ate, mu1_viz, mu0_viz, marginal_est, clf

        # Extrapolation Fitting
        m1.fit(X=X[mask1], Y=Y[mask1], T=T[mask1])
        m0.fit(X=X[mask0], Y=Y[mask0], T=T[mask0])

        # 4. Evaluation (Reweighted)
        mu1_gen = m1.predict_mean(X_eval=X_gen)
        mu0_gen = m0.predict_mean(X_eval=X_gen)

        e_gen = clf.predict_proba(X_gen)[:, 1]

        b1 = m1.predict_beta(X=X_gen, Y=mu1_gen, T=1.0)
        b0 = m0.predict_beta(X=X_gen, Y=mu0_gen, T=0.0)

        term1 = b1 * e_gen
        term0_e = 1 - e_gen
        term2 = b0 * term0_e
        b_total = term1 + term2

        diff = mu1_gen - mu0_gen
        val_diff = diff.flatten()

        denom_beta = b_total + 1e-6
        inv_beta = 1.0 / denom_beta
        weights = inv_beta.flatten()

        ate = np.average(val_diff, weights=weights)

        mu1_viz = None
        mu0_viz = None
        if X_viz is not None:
            mu1_viz = m1.predict_mean(X_eval=X_viz)
            mu0_viz = m0.predict_mean(X_eval=X_viz)

        return ate, mu1_viz, mu0_viz, marginal_est, clf

    def run_aipw(self, data, c=0.05, oracle=False):
        """
        AIPW / Doubly-Robust estimator restricted to the overlap region e ∈ [c, 1-c].
        oracle=True uses the full unselected population (X_pop, Y_pop, T_pop).
        """
        if oracle:
            X, Y, T = data['X_pop'], data['Y_pop'], data['T_pop'].astype(float)
        else:
            X, Y, T = data['X'], data['Y'], data['T'].astype(float)

        nn_clf = MLPClassifier((32,32), max_iter=500, random_state=self.seed)
        clf = CalibratedClassifierCV(nn_clf)
        clf.fit(X, T)
        e = clf.predict_proba(X)[:, 1]

        mask = (e >= c) & (e <= 1 - c)
        X, Y, T, e = X[mask], Y[mask], T[mask], e[mask]

        mu1 = LinearRegression().fit(X[T == 1], Y[T == 1]).predict(X)
        mu0 = LinearRegression().fit(X[T == 0], Y[T == 0]).predict(X)

        scores = (mu1 - mu0
                  + T * (Y - mu1) / e
                  - (1 - T) * (Y - mu0) / (1 - e))
        return float(np.mean(scores))

    def run_tmle(self, data, c=0.05):
        """
        TMLE (van der Laan & Rose, 2011) restricted to the overlap region e ∈ [c, 1-c].
        Applied naively to selected data — no selection-bias correction.
        """
        X, Y, T = data['X'], data['Y'], data['T'].astype(float)

        nn_clf = MLPClassifier((32,32), max_iter=500, random_state=self.seed)
        clf = CalibratedClassifierCV(nn_clf)
        clf.fit(X, T)
        e = clf.predict_proba(X)[:, 1]

        mask = (e >= c) & (e <= 1 - c)
        X, Y, T, e = X[mask], Y[mask], T[mask], e[mask]
        n = len(Y)

        # Initial outcome model Q(X, T)
        XT  = np.column_stack([X, T.reshape(-1, 1)])
        XT1 = np.column_stack([X, np.ones((n, 1))])
        XT0 = np.column_stack([X, np.zeros((n, 1))])
        Q_fit  = LinearRegression().fit(XT, Y)
        Q_pred = Q_fit.predict(XT)
        Q1     = Q_fit.predict(XT1)
        Q0     = Q_fit.predict(XT0)

        # Clever covariate + one-step fluctuation
        H   = T / e - (1 - T) / (1 - e)
        eps = LinearRegression(fit_intercept=False).fit(
            H.reshape(-1, 1), Y - Q_pred
        ).coef_[0]

        Q1_star = Q1 + eps / e
        Q0_star = Q0 - eps / (1 - e)
        return float(np.mean(Q1_star - Q0_star))

    def run_heckman(self, data):
        """
        Heckman (1979) two-step selection correction.
        Uses the full-population selection indicator S_pop from the data generator.
        Cannot perform well under non-Gaussian noise but included as a classical baseline.
        """
        X_pop = data['X_pop']
        S_pop = data['S_pop'].astype(float)
        X_obs, Y_obs, T_obs = data['X'], data['Y'], data['T'].astype(float)
        n_obs = len(Y_obs)

        # Step 1: Probit selection model P(S=1 | X) on full population
        sel_model = LogisticRegression(max_iter=1000).fit(X_pop, S_pop)
        z_obs = sel_model.decision_function(X_obs)

        # Inverse Mills Ratio: φ(z) / Φ(z)
        imr = norm.pdf(z_obs) / np.clip(norm.cdf(z_obs), 1e-8, 1.0)

        # Step 2: OLS  Y ~ 1 + X + T + IMR
        design = np.column_stack([np.ones(n_obs), X_obs, T_obs, imr])
        coef, *_ = np.linalg.lstsq(design, Y_obs, rcond=None)

        # ATE = coefficient on T: intercept(1) + X cols(x_dim) + T(1)
        x_dim = X_obs.shape[1] if X_obs.ndim > 1 else 1
        return float(coef[1 + x_dim])
