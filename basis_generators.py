#%%
import numpy as np
import pandas as pd
from scipy.stats import beta, t, lomax, norm, genpareto
from scipy.interpolate import BSpline


def standardize(df, threshold_tail=None, threshold_head=1e-8, check_positive=True, ref_stats=None, method='iqr'):
    """
    Standardize DataFrame using IQR and filter extreme columns.

    Args:
        df: Input DataFrame.
        ref_stats (dict): Optional. If provided, applies existing IQR and column mask.
                          Format: {'iqr': Series, 'cols': Index}
    Returns:
        (df_transformed, stats_dict)
    """
    if df.empty:
        return df, ref_stats

    if ref_stats is None:
        # === Training Mode ===
        if method == 'iqr':
            q75 = df.quantile(0.75)
            q25 = df.quantile(0.25)
            scale = q75 - q25
            scale[scale == 0] = 1.0
        elif method == 'max_abs':
            scale = df.abs().max()
            scale[scale == 0] = 1.0
        else:  # 'none'
            scale = pd.Series(1.0, index=df.columns)

        # 2. Temporary Normalize
        df_norm = df / scale

        # 3. Determine Valid Columns (Masking)
        if method == 'iqr':
            if threshold_tail is not None:
                mask_tail = df_norm.iloc[-1, :] < threshold_tail
            else:
                mask_tail = pd.Series([True] * df.shape[1], index=df.columns)

            if check_positive:
                mask_head = df_norm.iloc[0, :] > threshold_head
                mask = mask_tail & mask_head
            else:
                mask = mask_tail

            valid_cols = df.columns[mask]
        else:
            valid_cols = df.columns

        # 4. Final Filter
        df_final = df_norm[valid_cols]

        stats = {
            'scale': scale[valid_cols],
            'cols': valid_cols,
            'method': method
        }
        return df_final, stats

    else:
        # === TRANSFORM MODE (Testing) ===
        # 1. Align Columns
        common_cols = df.columns.intersection(ref_stats['scale'].index)

        df_norm = df.copy()

        # 2. Apply Scale
        if not common_cols.empty:
            df_norm[common_cols] = df[common_cols] / ref_stats['scale'][common_cols]

        # 3. Reindex to match training structure strictly (fill missing with 0)
        df_final = df_norm.reindex(columns=ref_stats['cols'], fill_value=0.0)

        return df_final, ref_stats


# ==============================================================================
# Basis Generators
# ==============================================================================

class GB2Basis:
    def __init__(self, a_grid, p_grid, q_grid):
        self.a_grid, self.p_grid, self.q_grid = a_grid, p_grid, q_grid

    def generate(self, u, ref_stats=None, standardize_output=True):
        data = {}
        for a in self.a_grid:
            for p in self.p_grid:
                for q in self.q_grid:
                    # 防止 beta 越界
                    z = np.clip(beta.ppf(u, p, q), 1e-9, 1-1e-9)
                    val = (z / (1.0 - z))**(1.0 / a)
                    data[f"GB2({a},{p},{q})"] = val

        if not standardize_output: return pd.DataFrame(data), None
        # GB2 is a positive distribution, need check_positive=True
        return standardize(
            pd.DataFrame(data), threshold_tail=1000, threshold_head=0.0001,
            ref_stats=ref_stats, check_positive=True, method='iqr'
        )


class LomaxBasis:
    def __init__(self, alpha_grid):
        self.alpha_grid = alpha_grid

    def generate(self, u, ref_stats=None, standardize_output=True):
        data = {}
        for a in self.alpha_grid:
            val = lomax.ppf(u, c=a)
            data[f"Lomax({a})"] = val

        if not standardize_output: return pd.DataFrame(data), None
        # Lomax is a positive distribution, need check_positive=True
        return standardize(pd.DataFrame(data), ref_stats=ref_stats, check_positive=True, method='iqr')


class SkewTBasis:
    def __init__(self, df_grid, gamma_grid):
        self.df_grid, self.gamma_grid = df_grid, gamma_grid

    def generate(self, u, ref_stats=None, standardize_output=True):
        data = {}
        for df in self.df_grid:
            for g in self.gamma_grid:
                qt = t.ppf(u, df)
                val = np.where(u <= 0.5, qt/g, qt*g)
                val -= np.median(val)
                data[f"SkewT({df},{g})"] = val

        if not standardize_output: return pd.DataFrame(data), None
        return standardize(pd.DataFrame(data), ref_stats=ref_stats, check_positive=False, method='iqr')


class NormalBasis:
    """
    Standard Normal Basis.
    """

    def __init__(self):
        pass

    def generate(self, u, ref_stats=None, standardize_output=True):
        data = {}
        val = norm.ppf(u)
        data["Normal"] = val

        if not standardize_output: return pd.DataFrame(data), None
        return standardize(pd.DataFrame(data), ref_stats=ref_stats, check_positive=False, method='iqr')


class StudentTBasis:
    """
    Student-t Basis.
    """

    def __init__(self, df_grid):
        self.df_grid = df_grid

    def generate(self, u, ref_stats=None, standardize_output=True):
        data = {}
        for df in self.df_grid:
            val = t.ppf(u, df)
            data[f"StudentT({df})"] = val

        if not standardize_output: return pd.DataFrame(data), None
        return standardize(pd.DataFrame(data), ref_stats=ref_stats, check_positive=False, method='iqr')


class GPDBasis:
    """
    Generalized Pareto Distribution (GPD) Basis.
    """

    def __init__(self, xi_grid):
        self.xi_grid = xi_grid

    def generate(self, u, ref_stats=None, standardize_output=True):
        data = {}
        for xi in self.xi_grid:
            # location=0, scale=1 default
            val = genpareto.ppf(u, c=xi)
            data[f"GPD({xi})"] = val

        if not standardize_output: return pd.DataFrame(data), None
        # GPD (with scale>0) defined at x >= 0，is a positive distribution, need check_positive=True
        return standardize(pd.DataFrame(data), ref_stats=ref_stats, check_positive=True, method='iqr')


class ISplineBasis:
    """
    I-Spline Basis Generator (Native Scipy Implementation).
    """

    def __init__(self, n_knots=5, order=3):
        self.n_knots = n_knots
        self.order = order
        self.degree = order - 1
        self.knots = None

    def _generate_knots(self):
        inner_knots = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        knots = np.concatenate(([0] * self.order,
                                inner_knots,
                                [1] * self.order))
        return knots

    def generate(self, u, ref_stats=None):
        """
        Generate the I-spline design matrix.
        Args:
            u: probability levels (p_levels)
            ref_stats: unused for Splines, kept for interface compatibility
        Returns:
            (pd.DataFrame, None) -> return format matches other generators
        """
        p_levels = np.array(u)

        # 1. Setup Knots
        if self.knots is None:
            self.knots = self._generate_knots()

        n_basis = len(self.knots) - self.order
        n_samples = len(p_levels)

        X_ispline = np.zeros((n_samples, n_basis))

        # 2. Compute I-splines
        for i in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0
            bs = BSpline(self.knots, coeffs, self.degree, extrapolate=False)

            t_start = self.knots[i]
            t_end = self.knots[i + self.order]
            span = t_end - t_start

            if span > 0:
                norm_factor = self.order / span
                bs_int = bs.antiderivative(1)
                X_ispline[:, i] = bs_int(p_levels) * norm_factor
            else:
                X_ispline[:, i] = 0.0

        X_ispline = np.clip(X_ispline, 0.0, 1.0)

        names = [f"ISpline_O{self.order}_K{self.n_knots}_{i}" for i in range(n_basis)]

        return pd.DataFrame(X_ispline, columns=names), None


# ==============================================================================
# 3. Segmented Wrapper
# ==============================================================================

class SegmentedGenerator:
    """
    Wraps any BasisGenerator to make it a local Tail Basis.
    Handles Mapping, Shifting, Masking, and Max-Abs Scaling.
    """

    def __init__(self, inner_generator, role='body', threshold=None, smooth=True):
        """
        Args:
            inner_generator: Instance of a Basis class (e.g., GPDBasis).
            role: 'body' (default), 'left', 'right'.
            threshold: Cutoff probability (e.g., 0.05 for left, 0.95 for right).
        """
        self.inner = inner_generator
        self.role = role
        self.smooth = smooth
        if threshold is None:
            self.thresholds = []
        elif np.isscalar(threshold):
            self.thresholds = [threshold]
        else:
            self.thresholds = list(threshold)

    def generate(self, u, ref_stats=None):
        u = np.array(u)
        n = len(u)

        df_parts = []

        for thresh in self.thresholds:

            # --- A. Coordinate Mapping (current thresh) ---
            u_mapped = np.zeros_like(u)
            mask_active = np.zeros(n, dtype=bool)

            if self.role == 'left':
                # u: 0 -> thresh  ==>  u': 1 -> 0
                mask_active = u < thresh
                if np.any(mask_active):
                    u_mapped[mask_active] = (thresh - u[mask_active]) / thresh

            elif self.role == 'right':
                # u: thresh -> 1  ==>  u': 0 -> 1
                mask_active = u > thresh
                if np.any(mask_active):
                    u_mapped[mask_active] = (u[mask_active] - thresh) / (1.0 - thresh)

            else:  # body
                u_mapped = u
                mask_active = np.ones(n, dtype=bool)

            if self.smooth and self.role in ['left', 'right']:
                u_mapped = u_mapped ** 2

            u_mapped = np.clip(u_mapped, 1e-9, 1 - 1e-9)

            # --- B. Call Inner Generator ---
            raw_df_part, _ = self.inner.generate(u_mapped, ref_stats=None, standardize_output=False)

            # --- C. Direction & Masking ---
            if self.role == 'left':
                raw_df_part = -raw_df_part

            for col in raw_df_part.columns:
                raw_df_part.loc[~mask_active, col] = 0.0

            # --- D. Renaming ---
            if self.role == 'left':
                thresh_str = f"{int(thresh * 100):02d}"
                prefix = f"L{thresh_str}_"  # e.g. L05_
            elif self.role == 'right':
                thresh_str = f"{int(thresh * 100):02d}"
                prefix = f"R{thresh_str}_"  # e.g. R95_
            else:
                prefix = ""

            raw_df_part.columns = [f"{prefix}{c}" for c in raw_df_part.columns]

            df_parts.append(raw_df_part)

        if not df_parts:
            return pd.DataFrame(index=range(n)), None

        full_raw_df = pd.concat(df_parts, axis=1)

        # --- E. Scaling (Max-Abs for Tails, IQR for Body) ---
        if ref_stats is None:
            # === Training ===
            if self.role in ['left', 'right']:
                return standardize(full_raw_df, method='max_abs', check_positive=False)
            else:
                return standardize(full_raw_df, method='iqr', check_positive=False)
        else:
            # === Testing ===
            return standardize(full_raw_df, ref_stats=ref_stats)


def build_design_matrix(n, generators, stats_list=None, u=None):
    if u is None:
        u = np.arange(1, n + 1) / (n + 1)  # (np.arange(n) + 0.5) / n
    df_list = []
    new_stats_list = []

    for i, gen in enumerate(generators):
        current_stat = stats_list[i] if stats_list else None

        df, stat = gen.generate(u, ref_stats=current_stat)

        df_list.append(df)
        new_stats_list.append(stat)

    X = pd.concat(df_list, axis=1)
    X.insert(0, "Intercept", 1.0)

    return X, new_stats_list


