import numpy as np

def generate_noise(n, noise_type, scale):
    """
    Generates centered noise of specific distribution type.
    """
    if noise_type == 'normal':
        return np.random.normal(0, scale, size=n)
    
    elif noise_type == 'laplace':
        # Heavier tails than normal
        return np.random.laplace(0, scale, size=n)
    
    elif noise_type == 'lognormal':
        # LogNormal is strictly positive. We generate and then subtract mean 
        # to make it zero-centered residual noise.
        # scale parameter here maps to sigma of the underlying normal
        raw_noise = np.random.lognormal(0, scale, size=n)
        return raw_noise - np.mean(raw_noise)
    
    elif noise_type == 'pareto':
        # Pareto is strictly positive and heavy tailed.
        # We use shape parameter a=3.0 (finite variance)
        a = 3.0
        raw_noise = np.random.pareto(a, size=n)
        # Center and scale
        # print out pareto for debugging
        print(f"Pareto Noise: {(raw_noise - np.mean(raw_noise)) * scale}")
        return (raw_noise - np.mean(raw_noise)) * scale
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def get_true_means(X=None):
    if X.ndim > 1 and X.shape[1] > 1:
        x_dim = X.shape[1]
        # Fixed coefficient vectors — normalized so scale matches the 1D case
        rng = np.random.default_rng(seed=0)
        A = rng.uniform(0.5, 1.5, size=x_dim) / x_dim   # linear
        B = rng.uniform(0.5, 1.5, size=x_dim) / x_dim   # quadratic
        C = 0.1 * rng.uniform(0.5, 1.5, size=x_dim) / x_dim  # quartic
        mu_0 = X @ A
        mu_1 = mu_0 + 1.0 + (X ** 2) @ B + (X ** 4) @ C
    else:
        X_flat = X[:, 0].flatten() if X.ndim > 1 else X.flatten()
        mu_0 = 1.0 * X_flat
        mu_1 = mu_0 + 1.0 + X_flat ** 2 + 0.1 * X_flat ** 4
    return mu_0, mu_1

def get_true_means_sinfunc(X=None):
    X_flat = X[:, 0].flatten() if X.ndim > 1 else X.flatten()
    # Linear baseline
    # mu_0 = 1.0 * X_flat + np.log(X_flat + 4.0)
    mu_0 = 2.0 * X_flat * np.sin(2.0 * X_flat)
    
    # Updated HTE function: mu_0 + 1 + X + 0.1*X^4
    # The X term adds linear divergence
    # The X^4 term adds massive divergence in tails
    term_const = 1.0
    # term_quartic = 0.1 * X_flat**2
    # term_quartic = (np.log(X_flat + 4.0))**2  #+ 0.1 *  X_flat**4
    
    term_quartic = X_flat**2 + 0.1 *  X_flat**4
    mu_1 = mu_0 + term_const + term_quartic
    return mu_0, mu_1

def get_true_means_logfunc(X=None):
    X_flat = X[:, 0].flatten() if X.ndim > 1 else X.flatten()
    mu_0 = np.log(X_flat + 4.0) * X_flat
    mu_1 = mu_0 + 1.0 + X_flat * X_flat * np.log(X_flat + 4.0) + 0.1 * (X_flat**4)
    return mu_0, mu_1

def generate_selection_bias_data(n=3000,
                                 x_dim=1,               # Dimensionality of covariate X
                                 x_threshold=2.0,       # Deterministic selection region (|x[:,0]| > threshold)
                                 beta_scale=3.0,        # Strength of non-deterministic selection
                                 beta_center=1.5,       # Center of non-deterministic selection
                                 noise_type='normal',   # 'normal', 'laplace', 'pareto', 'lognormal'
                                 noise_scale=0.5,
                                 binary_outcome=False,
                                 apply_sel_determin=False,
                                 apply_sel_non_determin=False,
                                 noise_func='additive',
                                 x_func='polynomial'):
    """
    Generates data with configurable selection bias and noise distributions.
    """
    # 1. Full Population Covariates
    X_pop = np.random.uniform(-3, 3, size=(n, x_dim))

    # Propensity for T generation — use first covariate
    if x_dim > 1:
        A_prop = np.random.uniform(0.5, 1.5, size=x_dim) / x_dim
        logits = X_pop @ A_prop
    else:
        logits = -(0.5 * X_pop[:, 0:1])
    propensity = 1 / (1 + np.exp(logits))
    T_pop = np.random.binomial(1, propensity).flatten()
    
    # 2. Outcomes (Functional form)
    if x_func == 'polynomial':
        mu_0_pop, mu_1_pop = get_true_means(X=X_pop)
    elif x_func == 'sin':
        mu_0_pop, mu_1_pop = get_true_means_sinfunc(X=X_pop)
    elif x_func == 'logfunc':
        mu_0_pop, mu_1_pop = get_true_means_logfunc(X=X_pop)

    # 3. Add Configurable Noise
    eps_0 = generate_noise(n, noise_type, noise_scale)
    eps_1 = generate_noise(n, noise_type, noise_scale)
    
    if noise_func == 'additive':
        y0_pop = mu_0_pop + eps_0
        y1_pop = mu_1_pop + eps_1
    elif noise_func == 'multiplicative':
        y0_pop = mu_0_pop * (1 + eps_0)
        y1_pop = mu_1_pop * (1 + eps_1)

    Y_pop = T_pop * y1_pop + (1 - T_pop) * y0_pop
    
    if binary_outcome:
        Y_pop = (Y_pop > np.median(Y_pop)).astype(float)
        
    # recalc mu_0_pop and mu_1_pop for binary case
    if binary_outcome:
        mu_0_pop = (mu_0_pop > np.median(Y_pop)).astype(float)
        mu_1_pop = (mu_1_pop > np.median(Y_pop)).astype(float)
    # 4. Apply Deterministic Selection
    # Drop Control units (T=0) in the tails defined by x_threshold
    X_obs = X_pop
    Y_obs = Y_pop
    T_obs = T_pop
    S_pop = np.ones(n, dtype=bool)

    if apply_sel_determin:
        X_obs, T_obs, Y_obs, S_det = apply_determine_selection(X_pop, T_pop, Y_pop, x_threshold)
        S_pop = S_det.copy()

    # 5. Apply Non-Deterministic Selection
    # Drop samples based on Y value using logistic selection function
    if apply_sel_non_determin:
        X_obs, T_obs, Y_obs, S_nondet = apply_nondeterm_selection(X_obs, T_obs, Y_obs, beta_scale, beta_center)
        obs_indices = np.where(S_pop)[0]
        S_pop[obs_indices[~S_nondet]] = False


    print(f"  > Generated {len(Y_obs)} samples after selection bias (from {n})")
    return {
        'X': X_obs.astype(np.float32),
        'Y': Y_obs.astype(np.float32),
        'T': T_obs.astype(np.float32),
        'X_pop': X_pop.astype(np.float32),
        'T_pop': T_pop.astype(np.float32),
        'Y_pop': Y_pop.astype(np.float32),
        'S_pop': S_pop,
        'mu_0_pop': mu_0_pop,
        'mu_1_pop': mu_1_pop,
        'true_ate': np.mean(mu_1_pop - mu_0_pop),
        'noise_info': f"{noise_type} (scale={noise_scale})"
    }

def apply_determine_selection(X_pop, T_pop, Y_pop, threshold):
    S = np.ones(len(X_pop), dtype=bool)
    # Selection bias: T=0 is missing if |X[:,0]| > threshold
    if X_pop.shape[1] > 1:
        A_sel_det = np.random.uniform(0.5, 1.5, size=X_pop.shape[1]) / X_pop.shape[1]
        mask_drop = (T_pop == 0) & (np.abs(X_pop @ A_sel_det) > threshold)
    else:
        mask_drop = (T_pop == 0) & (np.abs(X_pop[:, 0].flatten()) > threshold)
    S[mask_drop] = 0
    return X_pop[S], T_pop[S], Y_pop[S], S

def apply_nondeterm_selection(X, T, Y, scale, center):
    # Selection prob = sigmoid( scale * (Y - center) )
    if X.ndim > 1:
        A_sel_nondet = np.random.uniform(0.5, 1.5, size=X.shape[1]) / X.shape[1]
        logits = (Y + 0.1 * X @ A_sel_nondet - center) * scale
    else:
        logits = (Y + 0.1 * X[:, 0].flatten() - center) * scale
    prob_s = 1 / (1 + np.exp(-logits))

    selected = np.random.binomial(1, prob_s.flatten()).astype(bool)
    return X[selected], T[selected], Y[selected], selected