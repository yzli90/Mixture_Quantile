#%%
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time
import scipy.optimize

# Baseline imports
from scipy.stats import norm, lognorm, pareto, gamma, genpareto
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
from scipy.interpolate import interp1d


class QuantileMixture:
    def __init__(self, objective_type='MSE', env=None):
        """
        Gurobi backend.

        Args:
            objective_type (str): 'MSE' or 'MAE'.
        """
        self.objective_type = objective_type.upper()
        self.weights = None
        self.basis_info = None
        self.model_status = None
        self.solve_time = None
        self.fitted = False
        self.env = env

    def fit(self, X, y, sample_weight=None, basis_info=None, k=None, lambda_l1=0.0, lambda_sum_sq=0.0,
            sum_to_one=False, time_limit=60, verbose=0, M=100.0):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.array(sample_weight)
            if len(sample_weight) != n_samples:
                raise ValueError(f"sample_weight length ({len(sample_weight)}) must match n_samples ({n_samples})")
            if np.any(sample_weight < 0):
                raise ValueError("Sample weights must be non-negative.")

        if basis_info is not None:
            if hasattr(basis_info, '__len__') and len(basis_info) != n_features:
                raise ValueError(f"basis_info length ({len(basis_info)}) must match X columns ({n_features})")
            self.basis_info = basis_info.copy()
        else:
            self.basis_info = None

        is_spline = np.zeros(n_features, dtype=bool)
        is_intercept = np.zeros(n_features, dtype=bool)

        if self.basis_info is not None and 'name' in self.basis_info.columns:
            names = self.basis_info['name'].astype(str)
            is_spline = names.str.contains('ISpline', case=False).values
            is_intercept = (names == 'Intercept').values
        is_dist_basis = ~(is_spline | is_intercept)
        dist_indices = np.where(is_dist_basis)[0]

        # L1 regularization vector
        reg_l1_vec = np.zeros(n_features)
        reg_l1_vec[~is_spline] = lambda_l1  # <--- L1 for nonSpline

        # Spline smooth regularization matrix (Q_smooth)
        # obj: lambda * sum((w_{j+1} - w_j)^2)
        # quad w^T Q_smooth w
        Q_smooth = np.zeros((n_features, n_features))
        if lambda_sum_sq > 0 and np.any(is_spline):
            spline_indices = np.where(is_spline)[0]
            for i in range(len(spline_indices) - 1):
                idx_curr = spline_indices[i]
                idx_next = spline_indices[i + 1]

                # consecutive
                if idx_next == idx_curr + 1:
                    # (w_next - w_curr)^2 = w_next^2 - 2*w_next*w_curr + w_curr^2

                    # diagonal (+ w^2)
                    Q_smooth[idx_curr, idx_curr] += lambda_sum_sq
                    Q_smooth[idx_next, idx_next] += lambda_sum_sq

                    # intersection (- 2 * w_curr * w_next)
                    # (i, j) and (j, i), -lambda
                    Q_smooth[idx_curr, idx_next] -= lambda_sum_sq
                    Q_smooth[idx_next, idx_curr] -= lambda_sum_sq

        # Gurobi environment
        if self.env is None:
            env = gp.Env(empty=True)
            if not verbose:
                env.setParam("OutputFlag", 0)
            env.start()
            is_shared_env = False
        else:
            env = self.env
            is_shared_env = True

        model = gp.Model(env=env)
        model.setParam("TimeLimit", time_limit)

        # w >= 0
        w = model.addMVar(shape=n_features, lb=0.0, name="w")
        # w = model.addMVar(shape=n_features, lb=-GRB.INFINITY, name="w")

        if sum_to_one:
            model.addConstr(w.sum() == 1, name="sum_to_one")

        # Cardinality / L0 Norm
        if k is not None and len(dist_indices) > k:
            # distribution basis only
            z = model.addMVar(shape=len(dist_indices), vtype=GRB.BINARY, name="z_dist")

            model.addConstr(z.sum() <= k, name="dist_cardinality_limit")

            current_M = 1.0 if sum_to_one else M
            model.addConstr(w[dist_indices] <= current_M * z, name="big_m_dist_link")

        # Objective function
        # Lasso regularization: lambda * sum(w)
        reg_linear_term = reg_l1_vec @ w
        # reg_term = lambda_reg * (w**2).sum()

        if self.objective_type == 'MSE':
            # Weighted MSE: w' (X' S X) w - 2 (y' S X) w
            X_T_S = X.T * sample_weight
            Q_mse = X_T_S @ X
            c_mse = -2 * ((y * sample_weight) @ X)

            # Q(quad) = MSE Q + smooth Q
            Q_total = Q_mse + Q_smooth

            # linear = MSE linear + L1 regularization
            c_total = c_mse + reg_l1_vec

            model.setObjective(w @ Q_total @ w + c_total @ w, GRB.MINIMIZE)

        elif self.objective_type == 'MAE':
            # Weighted MAE: sum( s_i * u_i )
            u = model.addMVar(shape=n_samples, lb=0.0, name="u")

            # u = y - X @ w
            model.addConstr(u >= y - X @ w, name="abs_pos")
            model.addConstr(u >= X @ w - y, name="abs_neg")

            # obj: MAE + L1 + Spline smooth
            # (Quadratic Objective)
            mae_term = sample_weight @ u

            # MAE (linear) + L1 (linear) + Smooth (quad)
            model.setObjective(mae_term + reg_linear_term + w @ Q_smooth @ w, GRB.MINIMIZE)

        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")

        # print(np.linalg.lstsq(X, y, rcond=None)[0])

        model.optimize()
        self.solve_time = model.Runtime
        self.model_status = model.Status

        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            self.weights = w.X

            # clip
            self.weights[self.weights < 1e-8] = 0.0

            if sum_to_one and self.weights.sum() > 0:
                self.weights = self.weights / self.weights.sum()

            self.fitted = True

            if verbose:
                print(f"Optimization finished. Status: {model.Status}, Time: {self.solve_time:.4f}s")
        else:
            print(f"Warning: Optimization failed with status {model.Status}")
            self.weights = np.zeros(n_features)
            self.fitted = False

        if not is_shared_env:
            env.dispose()

        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted. Please call fit() first.")

        X = np.array(X)
        return X @ self.weights

    def get_active_basis(self):
        if not self.fitted:
            raise RuntimeError("Model not fitted")

        # active_idx = np.where(self.weights > 0)[0]
        active_idx = np.where(self.weights != 0)[0]

        result_data = {
            'index': active_idx,
            'weight': self.weights[active_idx]
        }
        result_df = pd.DataFrame(result_data)

        if self.basis_info is not None:
            if isinstance(self.basis_info, pd.DataFrame):
                subset_info = self.basis_info.iloc[active_idx].reset_index(drop=True)
                result_df = pd.concat([result_df, subset_info], axis=1)
            elif isinstance(self.basis_info, list) or isinstance(self.basis_info, np.ndarray):
                subset_info = [self.basis_info[i] for i in active_idx]
                result_df['basis_params'] = subset_info

        return result_df

    def pdf(self, y_eval, X_grid, p_grid):
        """
        Calculate Probability Density Function (PDF) values.
        f(y) = 1 / Q'(p) = dp / dy (via numerical differentiation)

        Args:
            y_eval (np.array): Points to evaluate density at.
            X_grid (np.array): Design matrix for a fine probability grid.
            p_grid (np.array): Probability levels for the grid.

        Returns:
            np.array: Density values at y_eval.
        """
        if not self.fitted: raise ValueError("Model not fitted")

        # 1. Predict Quantiles on the fine grid (The Shape)
        y_grid_pred = self.predict(X_grid)

        # 2. Numerical Differentiation
        dy = np.gradient(y_grid_pred)
        dp = np.gradient(p_grid)

        epsilon = 1e-12
        dy[dy < epsilon] = epsilon  # Avoid division by zero

        density_on_grid = dp / dy

        # 3. Interpolate to target points
        # Map: y -> density
        pdf_func = interp1d(y_grid_pred, density_on_grid, kind='linear',
                            bounds_error=False, fill_value=epsilon)

        density_vals = pdf_func(y_eval)

        # Clip to safe values
        density_vals[density_vals < epsilon] = epsilon
        return density_vals

    def log_likelihood(self, y_eval, X_grid, p_grid):
        """Calculate Total Log-Likelihood."""
        density = self.pdf(y_eval, X_grid, p_grid)
        return np.sum(np.log(density).clip(1e-12))


class GaussianMixtureMLE:

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, tol=1e-6, max_iter=1000)
        self.fitted = False
        self.solve_time = None

    def fit(self, y):
        """Fit GMM to the data y using MLE."""
        start_time = time.time()

        y = np.array(y).reshape(-1, 1)
        self.model.fit(y)

        self.solve_time = time.time() - start_time
        self.fitted = True
        return self

    def cdf(self, x):
        if not self.fitted: raise RuntimeError("Model not fitted")

        cdf_val = np.zeros_like(x, dtype=float)
        weights = self.model.weights_
        means = self.model.means_.flatten()
        covs = self.model.covariances_.flatten()

        for w, mean, cov in zip(weights, means, covs):
            std = np.sqrt(cov)
            cdf_val += w * norm.cdf(x, loc=mean, scale=std)
        return cdf_val

    # def predict(self, quantiles_levels):
    #     """
    #     predict Q(p). (Numerical Inversion).
    #
    #     Args:
    #         quantiles_levels (np.array): p, e.g. [0.01, ..., 0.99]
    #     """
    #     if not self.fitted: raise RuntimeError("Model not fitted")
    #
    #     quantiles_levels = np.array(quantiles_levels)
    #     results = []
    #
    #     low_bound = -1e7
    #     high_bound = 1e9
    #
    #     for p in quantiles_levels:
    #         try:
    #             # solve CDF(x) - p = 0
    #             q = brentq(lambda x: self.cdf(x) - p, low_bound, high_bound)
    #             results.append(q)
    #         except ValueError:
    #             results.append(np.nan)
    #
    #     return np.array(results)

    def predict(self, quantiles_levels, interp_grid_size=1000):
        if not self.fitted: raise RuntimeError("Not fitted")

        quantiles_levels = np.array(quantiles_levels)
        n_queries = len(quantiles_levels)

        means = self.model.means_.flatten()
        stds = np.sqrt(self.model.covariances_.flatten())
        low = (means - 10 * stds).min()
        high = (means + 10 * stds).max()

        def solve_one(p):
            try:
                return brentq(lambda x: self.cdf(x) - p, low, high)
            except ValueError:
                try:
                    return brentq(lambda x: self.cdf(x) - p, low - 100, high + 100)
                except:
                    return np.nan

        if n_queries < interp_grid_size:
            results = [solve_one(p) for p in quantiles_levels]
            return np.array(results)

        else:
            p_grid = np.linspace(1e-5, 1 - 1e-5, interp_grid_size)

            y_grid = np.array([solve_one(p) for p in p_grid])

            interpolator = interp1d(p_grid, y_grid, kind='linear', fill_value="extrapolate")

            return interpolator(quantiles_levels)

    def pdf(self, y_eval):
        """Return analytical PDF values."""
        if not self.fitted: raise RuntimeError("Not fitted")
        # score_samples returns log(density)
        log_prob = self.model.score_samples(y_eval.reshape(-1, 1))
        return np.exp(log_prob)

    def log_likelihood(self, y_eval):
        """Return Total Log-Likelihood."""
        if not self.fitted: raise RuntimeError("Not fitted")
        log_prob = self.model.score_samples(y_eval.reshape(-1, 1))
        return np.sum(log_prob)


class GPDMLE:
    """
    Generalized Pareto Distribution Estimator using MLE.
    Supports two modes:
    1. 'pot' (Peaks Over Threshold): Standard EVT approach. Fits GPD only to tail data > u.
    2. 'full': Fits GPD to the entire dataset (assuming data is strictly GPD distributed).
    """

    def __init__(self):
        self.mode = None
        self.params = {}  # Store xi, u, sigma, etc.
        self.y_train = None  # Store training data for empirical body estimation in POT
        self.valid_range = (0.0, 1.0)

    def fit(self, y, mode='pot', threshold=None, q_threshold=0.95, cd=None):
        """
        Args:
            y: 1D array of data (raw values, not just tail).
            mode: 'pot' or 'full'.
            threshold: (POT only) Absolute value of threshold u. If None, calculated from q_threshold.
            q_threshold: (POT only) Quantile to determine threshold (e.g. 0.95).
        """
        self.mode = mode
        self.y_train = np.sort(y)
        n_total = len(y)

        def my_gpd_optimizer(func, x0, args=(), **kwargs):
            return scipy.optimize.fmin(func, x0, args, ftol=1e-6, maxiter=1000, disp=False)

        if mode == 'pot':
            # 1. Determine Threshold (u)
            if threshold is None:
                # Use empirical quantile as threshold
                u = np.percentile(y, q_threshold * 100)
            else:
                u = threshold

            # 2. Extract Tail Data (Excesses)
            # data > u
            tail_data = y[y > u]
            n_exceed = len(tail_data)

            if n_exceed < 5:
                print(f"Warning: Too few tail samples ({n_exceed}) for GPD fit. Results may be unstable.")

            # 3. Fit GPD (Fix location = u)
            # scipy.stats.genpareto defines: (c, loc, scale) -> (xi, u, sigma)
            c_est, loc_est, scale_est = genpareto.fit(tail_data, floc=u, optimizer=my_gpd_optimizer)

            # 4. Store Parameters
            self.params = {
                'xi': c_est,  # Shape parameter
                'sigma': scale_est,  # Scale parameter
                'u': u,  # Threshold (Location)
                'phi': n_exceed / n_total,  # Tail probability P(X > u)
                'n_total': n_total,
                'n_exceed': n_exceed
            }

            if mode == 'pot':
                self.valid_range = (1 - self.params['phi'], 1.0)
            else:
                self.valid_range = (0.0, 1.0)
            return self

        elif mode == 'full':
            # Fit to entire dataset (Not recommended for financial returns usually)
            if not cd:
                c_est, loc_est, scale_est = genpareto.fit(y, optimizer=my_gpd_optimizer)
            else:
                c_est, loc_est, scale_est = genpareto.fit(cd, optimizer=my_gpd_optimizer)
            self.params = {
                'xi': c_est,
                'sigma': scale_est,
                'u': loc_est,  # For full mode, loc is estimated
                'phi': 1.0  # Uses 100% of data
            }

        return self

    def predict(self, p_levels):
        """
        Calculate Quantiles (VaR) for given probability levels.
        """
        p_levels = np.array(p_levels)
        res = np.zeros_like(p_levels)

        xi = self.params['xi']
        sigma = self.params['sigma']
        u = self.params['u']
        phi = self.params.get('phi', 1.0)

        if self.mode == 'pot':
            # Logic:
            # If p is in the tail (p > 1 - phi): Use GPD formula
            # If p is in the body (p <= 1 - phi): Use Empirical Quantile from training data

            # Threshold probability cutoff
            p_cutoff = 1 - phi

            # Mask for tail
            mask_tail = p_levels >= p_cutoff

            # A. Tail Part (GPD Formula)
            # VaR_p = u + (sigma/xi) * ( ((1-p)/phi)^(-xi) - 1 )
            # Note: (N/Nu) * (1-p) is equivalent to (1-p) / phi
            if np.any(mask_tail):
                tail_p = p_levels[mask_tail]
                term = ((1 - tail_p) / phi) ** (-xi)
                res[mask_tail] = u + (sigma / xi) * (term - 1)

            # B. Body Part (Empirical Quantile)
            if np.any(~mask_tail):
                # body_p = p_levels[~mask_tail]
                # # Use numpy percentile (linear interpolation)
                # res[~mask_tail] = np.percentile(self.y_train, body_p * 100)
                res[~mask_tail] = np.nan

        elif self.mode == 'full':
            # Use standard GPD PPF
            res = genpareto.ppf(p_levels, c=xi, loc=u, scale=sigma)

        return res

    def pdf(self, x):
        """
        Calculate Probability Density Function.
        """
        x = np.array(x)
        pdf_vals = np.zeros_like(x)

        xi = self.params['xi']
        sigma = self.params['sigma']
        u = self.params['u']
        phi = self.params.get('phi', 1.0)

        if self.mode == 'pot':
            # If x > u: f(x) = phi * g_gpd(x)
            # If x <= u: f(x) = 0 (or undefined in pure GPD context, we set 0 for plotting tail)
            mask_tail = x > u
            if np.any(mask_tail):
                # Calculate standard GPD pdf
                g_pdf = genpareto.pdf(x[mask_tail], c=xi, loc=u, scale=sigma)
                # Scale by tail ratio
                pdf_vals[mask_tail] = phi * g_pdf

        elif self.mode == 'full':
            pdf_vals = genpareto.pdf(x, c=xi, loc=u, scale=sigma)

        return pdf_vals


