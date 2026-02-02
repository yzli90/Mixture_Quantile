#%%
import numpy as np
import pandas as pd
from scipy.stats import kstest


class RiskMetrics:
    def __init__(self, y_true, y_pred_quantiles=None, quantiles_levels=None, valid_mask=None):
        """
        Risk metrics calculator tailored for tail risk and distribution fitting.

        Args:
            y_true (np.array): True observed samples (Testing set).
            y_pred_quantiles (np.array): Predicted quantiles from the model.
            quantiles_levels (np.array): Probability levels p corresponding to predictions.
        """
        self.y_true = np.sort(np.array(y_true))
        self.n = len(self.y_true)

        # If model predictions are provided
        if y_pred_quantiles is not None:
            self.y_pred = np.array(y_pred_quantiles)
        else:
            self.y_pred = None

        # Probability levels
        if quantiles_levels is not None:
            self.p_levels = np.array(quantiles_levels)
        else:
            # Default to empirical positions: (i + 0.5) / n
            self.p_levels = np.arange(1, self.n + 1) / (self.n + 1)

        # if valid_mask is not None:
        #     self.y_true = self.y_true[valid_mask]
        #     self.y_pred = self.y_pred[valid_mask]
        #     self.p_levels = self.p_levels[valid_mask]

        self.min_p = self.p_levels.min()
        self.max_p = self.p_levels.max()

    def get_tail_mask(self, threshold_q=0.95):
        """Get boolean mask for tail indices."""
        return self.p_levels >= threshold_q

    def mse(self, tail_threshold=None):
        """Mean Squared Error (Optional: Tail only)."""
        if self.y_pred is None: return np.nan
        diff = self.y_pred - self.y_true
        if tail_threshold is not None:
            mask = self.get_tail_mask(tail_threshold) & (~np.isnan(self.y_pred))
            if np.sum(mask) == 0: return np.nan
            return np.mean(diff[mask] ** 2)
        return np.mean(diff ** 2)

    def mae(self, tail_threshold=None):
        """Mean Absolute Error (Optional: Tail only)."""
        if self.y_pred is None: return np.nan
        diff = np.abs(self.y_pred - self.y_true)
        if tail_threshold is not None:
            mask = self.get_tail_mask(tail_threshold) & (~np.isnan(self.y_pred))
            if np.sum(mask) == 0: return np.nan
            return np.mean(diff[mask])
        return np.mean(diff)

    def w_distance(self):
        """Wasserstein Distance (Earth Mover's Distance)."""
        if self.y_pred is None: return np.nan
        return np.mean(np.abs(self.y_pred - self.y_true))

    def var_metric(self, alpha=0.95):
        """Value at Risk (VaR) comparison."""
        if self.y_pred is None or self.p_levels.max() < alpha: return np.nan, np.nan, np.nan
        idx = max((np.abs(self.p_levels - alpha)).argmin(), np.where(~np.isnan(self.y_pred))[0][0])
        var_pred = self.y_pred[idx]
        var_true = self.y_true[idx]
        return var_pred, var_true, np.abs(var_pred - var_true)

    def es_metric(self, alpha=0.95):
        """Expected Shortfall (ES) / CVaR comparison."""
        if self.y_pred is None: return np.nan, np.nan, np.nan
        mask = (self.p_levels >= alpha) & (~np.isnan(self.y_pred))
        if np.sum(mask) == 0: return np.nan, np.nan, np.nan
        es_pred = np.mean(self.y_pred[mask])
        es_true = np.mean(self.y_true[mask])
        return es_pred, es_true, np.abs(es_pred - es_true)

    def ks_stat(self):
        """Kolmogorov-Smirnov Statistic."""
        if self.y_pred is None: return np.nan
        if self.p_levels.min() < 0.1:
            stat, _ = kstest(self.y_true, lambda x: np.interp(x, self.y_pred, self.p_levels))
        else:
            u_pred = np.interp(self.y_true, self.y_pred, self.p_levels)
            stat = np.max(np.abs(self.p_levels - u_pred))
        return stat

    def report(self, model_name="Model", llh=None, tail_llh=None):
        """
        Generate a comprehensive report.

        Args:
            model_name (str): Name for the report column.
            llh (float, optional): Total Log-Likelihood calculated externally by the model.
        """
        is_full_distribution = (self.min_p < 0.01) and (self.max_p > 0.99)
        metrics = {}

        # --- A. Fit Quality ---
        metrics['Wasserstein (MAE)'] = self.w_distance()
        metrics['MSE'] = self.mse()
        metrics['KS_Stat'] = self.ks_stat()

        # --- B. Likelihood ---
        # We simply report what is passed to us
        metrics['LogLikelihood'] = llh if llh is not None else np.nan
        metrics['Tail_LogLikelihood'] = tail_llh if tail_llh is not None else np.nan

        if self.min_p <= 0.951 and self.max_p > 0.989:
            metrics['Tail_MAE_95'] = self.mae(tail_threshold=0.95)
            metrics['Tail_MSE_95'] = self.mse(tail_threshold=0.95)

            _, _, var_diff_95 = self.var_metric(0.95)
            _, _, es_diff_95 = self.es_metric(0.95)
            metrics['VaR_Diff_95'] = var_diff_95
            metrics['ES_Diff_95'] = es_diff_95

        if self.min_p <= 0.991 and self.max_p > 0.989:
            metrics['Tail_MAE_99'] = self.mae(tail_threshold=0.99)
            metrics['Tail_MSE_99'] = self.mse(tail_threshold=0.99)

            _, _, var_diff_99 = self.var_metric(0.99)
            _, _, es_diff_99 = self.es_metric(0.99)
            metrics['VaR_Diff_99'] = var_diff_99
            metrics['ES_Diff_99'] = es_diff_99

        return pd.Series(metrics, name=model_name)

