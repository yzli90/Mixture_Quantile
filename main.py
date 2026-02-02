#%%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
from basis_generators \
    import GB2Basis, LomaxBasis, SkewTBasis, NormalBasis, StudentTBasis, GPDBasis, SegmentedGenerator, ISplineBasis
from scipy.stats import norm, genpareto

from cross_validation import run_cv, aggregate_results
from report import RiskReporter

dataset = 'electricity'  # 'synthetic', 'synthetic_concat', 'gaussian', 'lendingclub', 'electricity', 'electricity_tail', 'pareto', 'sp500_ret'
output_dir = dataset
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
spline = True
if spline: n_knots = 10
left_tail, right_tail = False, True
card = False
if left_tail: left_thred = 0.25 + 0.01
if right_tail: right_thred = 0.95
obj, lambda_reg1, lambda_reg2, kfold = 'MSE', 0, 100, 5
gpd = True
# Body
generators = [
    # SkewTBasis(df_grid=[5, 10], gamma_grid=[0.8, 1.2]),
    # GPDBasis(xi_grid=[-2, -1, -0.5, 0, 0.5, 1.0, 2.0]),
    StudentTBasis(df_grid=[1, 5, 10, 30]),
    NormalBasis()
]
# generators = []
gaussian_components = 10

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
if dataset == 'synthetic_concat':
    n_samples = 2000
    # A. Body (Normal): 80% data, mean 8%，scale 2%
    y_norm = np.random.normal(loc=0.08, scale=0.02, size=int(0.8 * n_samples))
    # B. Tail (Pareto/Lomax): 20% data
    # Tail starting from 0.10 (10%) with shape=1
    y_tail = 0.10 + np.random.pareto(a=1.0, size=int(0.2 * n_samples)) * 0.05
    y_all = np.concatenate([y_norm, y_tail])
elif dataset == 'synthetic':
    n_samples = 2000
    u = (np.arange(n_samples) + 0.5) / n_samples
    # A: Normal Body
    # loc=0, scale=1
    q_norm = norm.ppf(u)
    # B: GPD Tail
    # shape(xi)=0.5, scale=1
    # GPD: ppf(u) = ( (1-u)^(-xi) - 1 ) / xi
    xi = 0.5
    q_gpd = ((1 - u)**(-xi) - 1) / xi
    y_synthetic = 0.08 + 0.02 * q_norm + 0.01 * q_gpd
    y_all = np.sort(y_synthetic)
elif dataset == 'gaussian':
    n_samples = 5000
    means = [10, 10, 12, 13, 14]
    stds = [0.2, 2.5, 2, 2, 2]
    weights = [0.3, 0.3, 0.1, 0.2, 0.1]  # mix weight
    np.random.seed(42)
    counts = np.random.multinomial(n_samples, weights)
    y_all = np.concatenate([
        np.random.normal(m, s, c) for m, s, c in zip(means, stds, counts)
    ])
    gaussian_components = len(means)
elif dataset == 'lendingclub':
    df = pd.read_csv('loan_2008.csv')
    y_all = df['int_rate'].dropna().values
    if np.mean(y_all) > 1.0: y_all = y_all / 100.0
elif 'electricity' in dataset:
    df = pd.read_csv('clean_data.csv')
    y_all = df['price'].dropna().values
    y_all = y_all[y_all != 0]
    if dataset == 'electricity_tail':
        y_all = y_all[y_all >= np.percentile(y_all, 95)]
elif dataset == 'pareto':
    n_samples = 2000
    xi = 0.5
    scale = 1.0
    loc = 0.0
    y_all = genpareto.rvs(c=xi, loc=loc, scale=scale, size=n_samples)
elif dataset == 'sp500_ret':
    df = pd.read_csv('sp500_ret.csv')
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index).date
    df = df['adjusted'].pct_change()
    y_all = df.dropna()
    y_all = y_all.loc[y_all.index <= datetime.date(2012, 12, 31)]

y_all = np.sort(y_all)
# -----------------------------------------------------------------------------
# 2. Generators
# -----------------------------------------------------------------------------
# generators = [
#     GB2Basis(a_grid=[1, 5], p_grid=[1, 2], q_grid=[1, 2]),
#     SkewTBasis(df_grid=[5, 10], gamma_grid=[0.8, 1.2]),
#     NormalBasis(),
#     StudentTBasis(df_grid=[5, 10, 30]),
#     GPDBasis(xi_grid=[-0.2, 0, 0.2, 0.5, 1.0, 2.0])
# ]

if spline: generators.append(ISplineBasis(n_knots=n_knots, order=3))
if left_tail:
    generators.append(SegmentedGenerator(
        GPDBasis(xi_grid=[-1, -0.5, -0.1, 0, 0.1, 0.5, 1.0]),
        role='left',
        threshold=list(np.round(np.arange(0.05, left_thred, 0.05), 2)),
        smooth=False
    ))
if right_tail:
    generators.append(SegmentedGenerator(
        GPDBasis(xi_grid=[-1, -0.5, 0, 0.5, 1.0]),
        role='right',
        threshold=list(np.round(np.arange(right_thred, 0.951, 0.05), 2)),
        smooth=False
    ))
# -----------------------------------------------------------------------------
# 3. Cross Validation
# -----------------------------------------------------------------------------
res_in, res_out, last_fold = run_cv(
    y_all, generators, k=kfold, objective_type=obj,
    lambda_reg1=lambda_reg1, lambda_reg2=lambda_reg2, gaussian_components=gaussian_components,
    card=card, gpd_include=gpd
)

# -----------------------------------------------------------------------------
# 4. Print Metrics Report
# -----------------------------------------------------------------------------
print("\n========== In-Sample Performance (Average) ==========")
res_in_aggr = aggregate_results(res_in)
print(res_in_aggr)

print("\n========== Out-of-Sample Performance (Average) ==========")
res_out_aggr = aggregate_results(res_out)
print(res_out_aggr)

# -----------------------------------------------------------------------------
# 5. Visualize Report
# -----------------------------------------------------------------------------
def get_sub_report(reporter, model_names):
    sub_preds = {name: reporter.preds[name] for name in model_names if name in reporter.preds}
    sub_densities = {}
    if reporter.densities:
        sub_densities = {name: reporter.densities[name] for name in model_names if name in reporter.densities}

    # Temp Reporter
    return RiskReporter(
        y_true=reporter.y_true,
        p_levels=reporter.p_levels,
        model_preds=sub_preds,
        model_densities=sub_densities
    )

is_single_l1 = isinstance(lambda_reg1, (int, float))

# --- A. In-Sample Plots ---
# print("\n>>> Generating In-Sample Plots...")
reporter_in = RiskReporter(
    y_true = last_fold['train']['y'],
    p_levels = last_fold['train']['p'],
    model_preds = last_fold['train']['preds'],
    model_densities = last_fold['train']['densities'],
    output_dir=output_dir
)
# reporter_in.plot_density(title="In-Sample Density")
# reporter_in.plot_quantile(title="In-Sample Quantile Plot")
# reporter_in.plot_zipf(title="In-Sample Zipf Plot")
# reporter_in.plot_qq(title="In-Sample QQ Plot")

# --- B. Out-of-Sample Plots ---
# print("\n>>> Generating Out-of-Sample Plots...")
reporter_out = RiskReporter(
    y_true = last_fold['test']['y'],
    p_levels = last_fold['test']['p'],
    model_preds = last_fold['test']['preds'],
    model_densities = last_fold['test']['densities'],
    output_dir=output_dir
)
# reporter_out.plot_density(title="Out-of-Sample Density")
# reporter_out.plot_quantile(title="Out-of-Sample Quantile Plot")
# reporter_out.plot_zipf(title="Out-of-Sample Zipf Plot")
# reporter_out.plot_qq(title="Out-of-Sample QQ Plot")

reporters = [('In-Sample', reporter_in), ('Out-of-Sample', reporter_out)]
for stage_name, base_reporter in reporters:
    group1_names = ['MQ', 'GMM', 'GPD']
    group2_names = [n for n in base_reporter.preds.keys() if '_L1_' in n or '_k' in n]

    if is_single_l1:
        base_reporter.plot_density(title=f"{stage_name} Density")
        base_reporter.plot_quantile(title=f"{stage_name} Quantile Plot")
        base_reporter.plot_zipf(title=f"{stage_name} Zipf Plot")
        base_reporter.plot_qq(title=f"{stage_name} QQ Plot")
    else:
        for i, group in enumerate([group1_names, group2_names], 1):
            if not group: continue
            sub_rep = get_sub_report(base_reporter, group)

            sub_rep.plot_density(title=f"{stage_name} Density")
            sub_rep.plot_quantile(title=f"{stage_name} Quantile Plot")
            sub_rep.plot_zipf(title=f"{stage_name} Zipf Plot")
            sub_rep.plot_qq(title=f"{stage_name} QQ Plot")
# --- C. Model Weights (MQ) ---
print("\n>>> Plotting MQ Active Basis...")
for name, m in last_fold['models'].items():
    if 'MQ' in name:
        print(f"Plotting {name}...")
        reporter_out.plot_active_basis(m, title=f"{name} Active Basis")

        q75, q25 = np.percentile(y_all, [75, 25])
        target_iqr = q75 - q25
        # reporter_out.plot_effective_weights(m, target_scale=target_iqr)

# weights = last_fold['models']['MQ'].get_active_basis()
res_in_aggr.to_csv(os.path.join(output_dir, "res_in.csv"))
res_out_aggr.to_csv(os.path.join(output_dir, "res_out.csv"))
