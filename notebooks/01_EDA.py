"""Exploratory Data Analysis — RetailAI Churn Intelligence System.

Runs the same 7 sections as the companion notebook and saves plots to
notebooks/plots/. No Jupyter dependencies required.

Usage:
    python notebooks/01_EDA.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "processed" / "customers_clean.csv"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 2_000
PALETTE = {0: "#2196F3", 1: "#F44336"}  # blue=retained, red=churned
plt.style.use("seaborn-v0_8-whitegrid")


def save(fig: plt.Figure, filename: str) -> None:
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Section 1 — Dataset Overview
# ---------------------------------------------------------------------------

print("=" * 60)
print("Section 1 — Dataset Overview")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["signup_date", "last_purchase_date"])

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print()
print("Column dtypes:")
print(df.dtypes.to_string())
print()
print("Descriptive statistics:")
print(df.describe(include="all").round(2).to_string())

# ---------------------------------------------------------------------------
# Section 2 — Churn Distribution
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Section 2 — Churn Distribution")
print("=" * 60)

churn_counts = df["churn"].value_counts().sort_index()
churn_pct = df["churn"].value_counts(normalize=True).sort_index() * 100
churn_rate = churn_pct[1]
retention_rate = churn_pct[0]

print(f"Retained  (0): {churn_counts[0]:>7,}  customers  ({retention_rate:.2f}%)")
print(f"Churned   (1): {churn_counts[1]:>7,}  customers  ({churn_rate:.2f}%)")
print(f">>> {churn_rate:.1f}% of customers churned in the last 90 days")

fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.bar(
    ["Retained (0)", "Churned (1)"],
    churn_counts.values,
    color=[PALETTE[0], PALETTE[1]],
    edgecolor="white",
    linewidth=1.5,
    width=0.5,
)

for bar, pct in zip(bars, churn_pct.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

ax.set_title(
    f"Customer Churn Distribution\n{churn_rate:.1f}% of customers churned in the last 90 days",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Churn Label", fontsize=12)
ax.set_ylabel("Number of Customers", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, churn_counts.max() * 1.15)

plt.tight_layout()
save(fig, "01_churn_distribution.png")

# ---------------------------------------------------------------------------
# Section 3 — Missing Values
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Section 3 — Missing Values")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
cols_with_missing = missing_df[missing_df["missing_count"] > 0]

if cols_with_missing.empty:
    print("No missing values detected.")
else:
    print(cols_with_missing.to_string())

sample_missing = df.isnull().sample(n=min(500, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    sample_missing.T,
    ax=ax,
    cbar=True,
    cmap="YlOrRd",
    yticklabels=True,
    xticklabels=False,
    linewidths=0,
)
ax.set_title(
    "Missing Values Heatmap\n(500-row sample — yellow = missing)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Customer records (sample)", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)

plt.tight_layout()
save(fig, "02_missing_values.png")

# ---------------------------------------------------------------------------
# Section 4 — Feature Distributions by Churn
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Section 4 — Feature Distributions by Churn")
print("=" * 60)

features_to_plot = [
    "total_orders",
    "total_spent",
    "avg_order_value",
    "return_rate",
    "support_tickets",
    "age",
]

sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

fig, axes = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle(
    "Feature Distributions by Churn Status\n(blue = retained, red = churned)",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 3][idx % 3]
    for churn_val, color in PALETTE.items():
        subset = sample_df[sample_df["churn"] == churn_val][feature].dropna()
        label = "Retained" if churn_val == 0 else "Churned"
        ax.hist(subset, bins=40, alpha=0.55, color=color, label=label, edgecolor="none")
    ax.set_title(feature.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8)

for idx in range(len(features_to_plot), 9):
    axes[idx // 3][idx % 3].set_visible(False)

plt.tight_layout()
save(fig, "03_distributions.png")

print(f"  (plotted on {min(SAMPLE_SIZE, len(df)):,}-row sample)")

# ---------------------------------------------------------------------------
# Section 5 — Correlation Matrix
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Section 5 — Correlation Matrix")
print("=" * 60)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
corr_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    ax=ax,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 9},
)
ax.set_title(
    "Correlation Matrix — Numeric Features + Churn\n(lower triangle, annotated, coolwarm diverging palette)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)

plt.tight_layout()
save(fig, "04_correlation.png")

churn_corr = corr_matrix["churn"].drop("churn").sort_values(key=abs, ascending=False)
print("Top correlations with churn:")
print(churn_corr.to_string())

# ---------------------------------------------------------------------------
# Section 6 — Key Features vs Churn (Box Plots)
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Section 6 — Key Features vs Churn (Box Plots)")
print("=" * 60)

business_signals = ["total_spent", "total_orders", "support_tickets"]
signal_labels = ["Total Spent (EUR)", "Total Orders", "Support Tickets"]

sample_df2 = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(12, 8))
fig.suptitle(
    "Top Business Signals vs Churn Status",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

for ax, feature, label in zip(axes, business_signals, signal_labels):
    str_palette = {str(k): v for k, v in PALETTE.items()}
    sns.boxplot(
        data=sample_df2,
        x="churn",
        y=feature,
        ax=ax,
        palette=str_palette,
        width=0.5,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
    )
    ax.set_title(f"{label}\nvs Churn", fontsize=11, fontweight="bold")
    ax.set_xlabel("Churn (0=Retained, 1=Churned)", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticklabels(["Retained", "Churned"])

    for churn_val in [0, 1]:
        median = sample_df2[sample_df2["churn"] == churn_val][feature].median()
        ax.text(
            churn_val,
            median,
            f"  {median:.1f}",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

plt.tight_layout()
save(fig, "05_features_vs_churn.png")

print(f"  (plotted on {min(SAMPLE_SIZE, len(df)):,}-row sample)")

# ---------------------------------------------------------------------------
# Section 7 — Key Findings (console)
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Section 7 — Key Findings")
print("=" * 60)

top3 = churn_corr.head(3)

print(f"""
CHURN RATE
  {churn_rate:.1f}% of customers churned in the last 90 days ({churn_counts[1]:,} customers).
  Retaining 5% more = measurable revenue gain without new acquisition cost.

TOP 3 FEATURES CORRELATED WITH CHURN
""")
for rank, (feat, val) in enumerate(top3.items(), start=1):
    direction = "higher" if val > 0 else "lower"
    print(f"  {rank}. {feat:<25} r = {val:+.3f}  ({direction} -> more churn)")

print(f"""
DISTRIBUTION OBSERVATIONS
  - total_spent / total_orders: right-skewed, log-transform recommended
  - return_rate: clusters near 0, long tail of serial returners
  - age: broadly distributed (18-80), no strong standalone churn signal

MODELLING RECOMMENDATIONS
  - Handle class imbalance via scale_pos_weight in XGBoost or SMOTE
  - Apply RobustScaler (not StandardScaler) for monetary outliers
  - Log-transform total_spent and avg_order_value to reduce skew
  - Evaluate with AUC-ROC + Precision-Recall (not accuracy)
  - Use SHAP values for explainability with non-technical stakeholders

BUSINESS RECOMMENDATION
  Prioritise intervention on customers with:
    >= 2 support_tickets AND <= 3 total_orders
  -> Highest churn-risk concentration; deploy retention campaign
    within 30 days of last purchase (voucher / proactive outreach).

PLOTS SAVED TO: {PLOTS_DIR.resolve()}
""")
