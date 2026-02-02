#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import os
from metrics import RiskMetrics


class RiskReporter:
    def __init__(self, y_true, p_levels, model_preds, model_densities=None, output_dir="."):
        """
        Args:
            model_preds: dict, {'MQ': y_pred_qm, 'GMM': y_pred_gmm}
            model_densities: dict, (Optional)
                             {'MQ': (y_vals, density_vals), 'GMM': (y_vals, density_vals)}
        """
        self.y_true = y_true
        self.p_levels = p_levels
        self.preds = model_preds
        self.densities = model_densities
        self.output_dir = output_dir
        self.colors = {'MQ': 'blue', 'GMM': 'red', 'GPD': 'green'}
        self.styles = {'MQ': '-', 'GMM': '--', 'GPD': '-.'}

    def _save_and_show(self, title):
        clean_title = title.replace(" ", "_").replace("$", "").replace("\\", "").replace("{", "").replace("}",
                                                                                                          "").replace(
            "=", "").lower()
        filename = f"{clean_title}.pdf"
        full_path = os.path.join(self.output_dir, filename)

        plt.tight_layout()
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Figure saved to: {full_path}")
        plt.show()

    def generate_metrics_table(self, llh_dict=None, times_dict=None):
        results = []
        for name, y_pred in self.preds.items():
            llh = llh_dict.get(name) if llh_dict else None
            m = RiskMetrics(self.y_true, y_pred, self.p_levels).report(name, llh=llh)

            if times_dict and name in times_dict:
                m['Training Time (s)'] = times_dict[name]

            results.append(m)

        return pd.concat(results, axis=1)

    def plot_density(self, title="Density Plot"):
        if not self.densities:
            print("No density data provided.")
            return

        plt.figure(figsize=(8, 6))

        # 1. Histogram
        plt.hist(self.y_true, bins=50, density=True, color='gray', alpha=0.3, label='True Hist')

        for name, (y_grid, dens_vals) in self.densities.items():
            mask = (y_grid >= self.y_true.min()) & (y_grid <= self.y_true.max())
            plt.plot(y_grid[mask], dens_vals[mask],
                     color=self.colors.get(name),
                     linestyle=self.styles.get(name),
                     linewidth=2, label=f'{name}')

        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()
        self._save_and_show(title)

    def plot_zipf(self, title="Zipf Plot (Log-Log)"):
        plt.figure(figsize=(8, 6))

        safe = (1 - self.p_levels) > 0
        x_true = np.log10(self.y_true[safe])
        y_surv = np.log10(1 - self.p_levels[safe])

        plt.scatter(x_true, y_surv, c='k', alpha=0.3, s=15, label='True Data')

        for name, y_pred in self.preds.items():
            x_pred = np.log10(y_pred[safe])
            plt.plot(x_pred, y_surv,
                     color=self.colors.get(name),
                     linestyle=self.styles.get(name),
                     linewidth=2, label=name)

        plt.title(title)
        plt.xlabel("log10(Value)")
        plt.ylabel("log10(Survival Probability)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()
        self._save_and_show(title)

    def plot_qq(self, title="QQ Plot"):
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        all_preds = np.concatenate([p for p in self.preds.values()])
        combined_min = min(self.y_true.min(), all_preds.min())
        combined_max = max(self.y_true.max(), all_preds.max())

        buffer = (combined_max - combined_min) * 0.05
        plot_min = combined_min - buffer
        plot_max = combined_max + buffer

        for name, y_pred in self.preds.items():
            ax.scatter(self.y_true, y_pred,
                       color=self.colors.get(name),
                       linestyle=self.styles.get(name),
                       s=15, alpha=0.5, label=name)

        ax.plot([plot_min, plot_max], [plot_min, plot_max],
                'k--', alpha=0.7, linewidth=1.5, label='45 degree line')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        plt.title(title)
        plt.xlabel("True quantiles")
        plt.ylabel("Predicted quantiles")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()
        self._save_and_show(title)

    def plot_quantile(self, title="Quantile Plot"):
        plt.figure(figsize=(8, 6))

        # 1. True Data
        plt.plot(self.p_levels, self.y_true,
                 color='gray', alpha=0.6, linewidth=3, label='True Data')

        for name, y_pred in self.preds.items():
            plt.plot(self.p_levels, y_pred,
                     color=self.colors.get(name),
                     linestyle=self.styles.get(name),
                     linewidth=2, label=name)

        plt.title(title)
        plt.xlabel("Probability Level $p$")
        plt.ylabel("Quantile Value $Q(p)$")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # plt.ylim(self.y_true.min(), self.y_true.max() * 1.1)

        plt.show()
        self._save_and_show(title)

    def plot_active_basis(self, model, top_k=10, title="Top Active Basis"):
        if not hasattr(model, 'get_active_basis'):
            return

        active = model.get_active_basis()
        if active.empty:
            print("No active basis found.")
            return

        if 'name' not in active.columns:
            if 'index' in active.columns:
                active['name'] = active['index'].astype(str)
            else:
                active['name'] = [f"Basis_{i}" for i in range(len(active))]

        if 'abs_weight' not in active.columns:
            active['abs_weight'] = active['weight'].abs()

        active = active[active['abs_weight'] >= 0.00005]
        plot_data = active.sort_values('abs_weight', ascending=False).head(top_k)

        plt.figure(figsize=(10, 6))

        colors = ['red' if w < 0 else 'blue' for w in plot_data['weight']]

        sns.barplot(x='weight', y='name', data=plot_data, palette=colors)

        plt.title(title)
        plt.xlabel("Basis Weight (Normed)")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_effective_weights(self, model, target_scale, top_k=15, title="Effective Coefficients"):
        """
        Effective Weight = Model_Weight * (Target_Scale / Basis_Scale)
        """
        if not hasattr(model, 'get_active_basis'): return

        # Get Basis Scale
        active = model.get_active_basis()  # include 'weight', 'name', 'scale'
        if active.empty: return

        if 'name' not in active.columns:
            if 'index' in active.columns:
                active['name'] = active['index'].astype(str)
            else:
                active['name'] = [f"Basis_{i}" for i in range(len(active))]

        if 'scale' not in active.columns:
            active['scale'] = 1.0

        # w_real = w_norm * S_y / S_x
        active['effective_weight'] = active['weight'] * (target_scale / active['scale'])

        plt.figure(figsize=(10, 6))

        active['abs_eff'] = active['effective_weight'].abs()
        plot_data = active.sort_values('abs_eff', ascending=False).head(top_k)

        colors = ['red' if x < 0 else 'blue' for x in plot_data['effective_weight']]

        sns.barplot(x='effective_weight', y='name', data=plot_data, palette=colors)

        plt.title(f"{title} (Impact on Y)")
        plt.xlabel("Effective Coefficient value (Real Scale)")
        plt.grid(True, alpha=0.3)
        plt.show()
        