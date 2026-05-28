import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import norm


def aipw_ate(data):
    """
    Doubly-Robust / AIPW estimator applied naively to selected (observed) data.
    Treats the selected subpopulation as the population — no selection correction.
    """
    X, Y, T = data['X'], data['Y'], data['T'].astype(float)

    mu1 = LinearRegression().fit(X[T == 1], Y[T == 1]).predict(X)
    mu0 = LinearRegression().fit(X[T == 0], Y[T == 0]).predict(X)
    e = np.clip(
        LogisticRegression(max_iter=500).fit(X, T).predict_proba(X)[:, 1],
        0.01, 0.99,
    )
    scores = (mu1 - mu0
              + T * (Y - mu1) / e
              - (1 - T) * (Y - mu0) / (1 - e))
    return float(np.mean(scores))


def tmle_ate(data):
    """
    TMLE (van der Laan & Rose, 2011) applied naively to selected data.
    Produces an efficient estimate of ATE within the selected subpopulation —
    no correction for how units entered the sample.
    """
    X, Y, T = data['X'], data['Y'], data['T'].astype(float)
    n = len(Y)

    # Step 1: Initial outcome model Q(X, T)
    XT  = np.column_stack([X, T.reshape(-1, 1)])
    XT1 = np.column_stack([X, np.ones((n, 1))])
    XT0 = np.column_stack([X, np.zeros((n, 1))])
    Q_fit  = LinearRegression().fit(XT, Y)
    Q_pred = Q_fit.predict(XT)
    Q1     = Q_fit.predict(XT1)
    Q0     = Q_fit.predict(XT0)

    # Step 2: Propensity g(X) = P(T=1|X)
    g = np.clip(
        LogisticRegression(max_iter=500).fit(X, T).predict_proba(X)[:, 1],
        0.01, 0.99,
    )

    # Step 3: Clever covariate + fluctuation (one-step, continuous outcome)
    H   = T / g - (1 - T) / (1 - g)
    eps = LinearRegression(fit_intercept=False).fit(
        H.reshape(-1, 1), Y - Q_pred
    ).coef_[0]

    # Step 4: Targeted update
    Q1_star = Q1 + eps / g
    Q0_star = Q0 - eps / (1 - g)
    return float(np.mean(Q1_star - Q0_star))


def heckman_ate(data):
    """
    Heckman (1979) two-step selection correction.
    Assumes a probit model for selection P(S=1|X) and jointly Gaussian errors.
    Uses the full-population selection indicator S_pop returned by the data generator.
    Cannot perform well under non-Gaussian noise but included as a classical baseline.
    """
    X_pop = data['X_pop']
    S_pop = data['S_pop'].astype(float)
    X_obs, Y_obs, T_obs = data['X'], data['Y'], data['T'].astype(float)
    n_obs = len(Y_obs)

    # Step 1: Probit selection model P(S=1 | X) on full population
    # Use logistic regression as a probit approximation (decision_function → z-score proxy)
    sel_model = LogisticRegression(max_iter=1000).fit(X_pop, S_pop)
    z_obs = sel_model.decision_function(X_obs)   # linear predictor for selected units

    # Inverse Mills Ratio: φ(z) / Φ(z)
    imr = norm.pdf(z_obs) / np.clip(norm.cdf(z_obs), 1e-8, 1.0)

    # Step 2: OLS  Y ~ 1 + X + T + IMR   (constant treatment-effect specification)
    X_flat = X_obs.flatten()
    design = np.column_stack([np.ones(n_obs), X_flat, T_obs, imr])
    coef, *_ = np.linalg.lstsq(design, Y_obs, rcond=None)

    # ATE = coefficient on T under constant-effect assumption
    return float(coef[2])
