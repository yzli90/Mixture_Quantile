#%%
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d

from basis_generators import build_design_matrix
from models import QuantileMixture, GaussianMixtureMLE, GPDMLE
from metrics import RiskMetrics


def run_cv(y_all, generators, k=5, objective_type='MAE', seed=50, lambda_reg1=0.0, lambda_reg2=0.0, max_basis_k=None,
           gaussian_components=10, card=False, gpd_include=False):
    l1_list = [lambda_reg1] if np.isscalar(lambda_reg1) else lambda_reg1
    obj_list = [objective_type] if isinstance(objective_type, str) else objective_type
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    results_in = []
    results_out = []
    last_fold_info = {}

    # shared_env = gp.Env(empty=True)
    # shared_env.setParam("OutputFlag", 0)
    # shared_env.start()

    print(f"Starting {k}-Fold Cross Validation...")

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(y_all)):
        print(f"  > Processing Fold {fold_i + 1}/{k}...", end="")

        # 1. Data Prep
        y_train = np.sort(y_all[train_idx])
        y_test = np.sort(y_all[test_idx])

        q75, q25 = np.percentile(y_train, [75, 25])
        scale = q75 - q25
        if scale == 0: scale = 1.0
        y_train_norm = y_train / scale

        # 2. Build Matrix
        X_train, train_stats = build_design_matrix(len(y_train), generators, stats_list=None)
        X_test, _ = build_design_matrix(len(y_test), generators, stats_list=train_stats)
        # train_stats is a list of dicts {'scale': Series, 'cols': Index, ...}

        # Capture Scales
        scale_list = []
        if train_stats:
            for s in train_stats:
                if s and 'scale' in s:
                    scale_list.append(s['scale'])

        if scale_list:
            full_scale = pd.concat(scale_list)
        else:
            full_scale = pd.Series(1.0, index=X_train.columns)

        # Build Basis Info
        basis_info = pd.DataFrame({
            'name': full_scale.index,
            'scale': full_scale.values
        })
        basis_info = basis_info.set_index('name').reindex(X_train.columns).reset_index()
        if 'index' in basis_info.columns:
            basis_info = basis_info.rename(columns={'index': 'name'})
        basis_info['scale'] = basis_info['scale'].fillna(1.0)

        models_dict = {}
        times_dict = {}

        # 3. Fit MQ
        for obj in obj_list:
            obj_tag = obj.upper()
            for l1 in l1_list:

                if len(l1_list) == 1 or l1 == 0:
                    # m_name = f"MQ ({obj_tag})"
                    m_name = f"MQ"
                else:
                    # m_name = f"MQ ({obj_tag}, $\lambda_1 = {l1}$)"
                    m_name = f"MQ ($\lambda_1 = {l1}$)"

                qm = QuantileMixture(objective_type=obj)
                qm.fit(X_train, y_train_norm,
                       basis_info=basis_info,
                       lambda_l1=l1,
                       lambda_sum_sq=lambda_reg2,
                       k=max_basis_k,
                       verbose=0)

                models_dict[m_name] = qm
                times_dict[m_name] = qm.solve_time

            if card:
                for k_val in [1, 2]:
                    # m_name = f"MQ: ({obj_tag}, $k \le {k_val}$)"
                    m_name = f"MQ: ($k \le {k_val}$)"

                    m_card = QuantileMixture(objective_type=obj)
                    m_card.fit(X_train, y_train_norm, basis_info=basis_info, lambda_l1=0,
                               lambda_sum_sq=lambda_reg2, k=k_val, verbose=0)
                    models_dict[m_name], times_dict[m_name] = m_card, m_card.solve_time

        # 4. Fit GMM
        gmm = GaussianMixtureMLE(n_components=gaussian_components)
        t0 = time.time()
        gmm.fit(y_train)
        times_dict['GMM'] = time.time() - t0
        models_dict['GMM'] = gmm

        # Fit GPD
        if gpd_include:
            gpd = GPDMLE()
            t0 = time.time()
            gpd.fit(y_train, mode='pot', q_threshold=0.95)
            # gpd.fit(y_train, mode='full')
            times_dict['GPD'] = time.time() - t0
            models_dict['GPD'] = gpd

        # 5. Density Grid (Global Model Curve)
        p_dense = np.concatenate([
            np.logspace(-6, -3, 100),  # left tail: 1e-6 to 1e-3
            np.linspace(0.001, 0.999, 1800),  # body
            1 - np.logspace(-6, -3, 100)[::-1]  # right tail
        ])
        n_dense = len(p_dense)
        X_dense, _ = build_design_matrix(n_dense, generators, stats_list=train_stats, u=p_dense)
        y_grid_plot = np.linspace(y_all.min(), y_all.max(), 1000)

        densities_plot = {}
        for name, m in models_dict.items():
            if isinstance(m, QuantileMixture):
                y_d_real = m.predict(X_dense) * scale
                d_d_real = m.pdf(y_d_real / scale, X_dense, p_dense) / scale
                densities_plot[name] = (y_d_real, d_d_real)
            else:
                densities_plot[name] = (y_grid_plot, m.pdf(y_grid_plot))

        # --- Internal Evaluate Function ---
        def evaluate(y_true, X_input, p_lvls):
            reports, preds = [], {}
            tail_mask = p_lvls >= 0.95
            mask = None
            for name, m in models_dict.items():
                if isinstance(m, QuantileMixture):
                    pred = m.predict(X_input) * scale
                    y_grid, dens = densities_plot[name]
                    interp = interp1d(y_grid, dens, kind='linear', bounds_error=False, fill_value=1e-12)
                    pdf_vals = interp(y_true).clip(1e-12)
                    llh = np.mean(np.log(pdf_vals))
                elif isinstance(m, GPDMLE):
                    pred = m.predict(p_lvls)
                    pdf_vals = m.pdf(y_true).clip(1e-12)
                    if m.mode == 'pot':
                        cutoff = 1 - m.params['phi']
                        mask = p_lvls >= cutoff
                        llh = np.nan
                    else:
                        llh = np.mean(np.log(pdf_vals))
                else:  # GMM
                    pred = m.predict(p_lvls)
                    pdf_vals = m.pdf(y_true).clip(1e-12)
                    llh = m.log_likelihood(y_true) / len(y_true)

                tail_llh = np.mean(np.log(pdf_vals[tail_mask]))

                preds[name] = pred
                rep = RiskMetrics(y_true, pred, p_lvls, valid_mask=mask).report(name, llh=llh, tail_llh=tail_llh)
                rep['Training Time (s)'] = times_dict[name]
                reports.append(rep)
            return pd.concat(reports, axis=1), preds

        res_in_fold, pr_in = evaluate(y_train, X_train, np.arange(1, len(y_train) + 1) / (len(y_train) + 1))
        res_out_fold, pr_out = evaluate(y_test, X_test, np.arange(1, len(y_test) + 1) / (len(y_test) + 1))

        results_in.append(res_in_fold)
        results_out.append(res_out_fold)

        if fold_i == k - 1:
            last_fold_info = {
                'models': models_dict,
                'train': {'y': y_train, 'p': np.arange(1, len(y_train) + 1) / (len(y_train) + 1), 'preds': pr_in,
                          'densities': densities_plot},
                'test': {'y': y_test, 'p': np.arange(1, len(y_test) + 1) / (len(y_test) + 1), 'preds': pr_out,
                         'densities': densities_plot}
            }
        print(f" Done.")

    return results_in, results_out, last_fold_info


def aggregate_results(results_list):
    all_models = results_list[0].columns
    idx = results_list[0].index
    df_avg = pd.DataFrame(index=idx)
    for m in all_models:
        m_vals = np.array([res[m].values for res in results_list])
        if 'Training Time (s)' in idx:
            time_idx_pos = list(idx).index('Training Time (s)')

            times = np.array([res[m].iloc[time_idx_pos] for res in results_list])
            avg_time = np.nanmean(times[1:]) if len(times) > 1 else times[0]

            other_avgs = np.nanmean(m_vals, axis=0)
            other_avgs[time_idx_pos] = avg_time
            df_avg[m] = other_avgs
        else:
            df_avg[m] = np.nanmean(m_vals, axis=0)

    if 'Training Time (s)' in idx:
        time_row = df_avg.loc[['Training Time (s)']]
        df_avg = pd.concat([time_row, df_avg.drop('Training Time (s)')])
    return df_avg




