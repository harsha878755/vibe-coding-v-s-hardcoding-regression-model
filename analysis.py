"""
Full statistical machine learning workflow for the Vibe Coding vs Hardcoding
dataset, mirroring the TMDB 5000 reference report's five-module structure:
  1. EDA
  2. Bootstrap resampling / confidence interval
  3. Hypothesis testing (two-sample Welch's t-test)
  4. Multiple linear regression (+ diagnostics)
  5. Generalized Linear Model (logistic regression) with odds ratios
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
sns.set_style("whitegrid")

df = pd.read_csv("../data/vibe_coding_vs_hardcoding.csv")
print("Shape:", df.shape)
print(df.dtypes)

PURPLE = "#7C5CFC"
TEAL = "#1E9E85"
PAL = {"Vibe Coding": PURPLE, "Hardcoding": TEAL}

# =========================================================
# MODULE 1: EXPLORATORY DATA ANALYSIS
# =========================================================

# --- descriptive stats table ---
num_cols = ["Time_Taken_Min", "Bugs_Found", "Code_Quality_Score",
            "Satisfaction_Score", "Lines_of_Code", "Debugging_Time_Min"]
desc = df[num_cols].agg(["mean", "std", "min", "median", "max"]).round(2)
desc.to_csv("../results/stats_overall_describe.csv")
print("\nOverall descriptive stats:\n", desc)

group_desc = df.groupby("Method")[num_cols].mean().round(2)
group_desc.to_csv("../results/stats_group_means.csv")
print("\nGroup means:\n", group_desc)

# Figure 5.1: Task count by Developer Experience & Method (grouped bar)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Developer_Experience", hue="Method",
              order=["Beginner", "Intermediate", "Expert"],
              palette=PAL)
plt.title("Figure 5.1: Task Count by Developer Experience and Method")
plt.xlabel("Developer Experience")
plt.ylabel("Number of Tasks")
plt.tight_layout()
plt.savefig("../images/fig5_1_experience_method_count.png", dpi=150)
plt.close()

# Figure 5.2: Crosstab heatmap - Task Complexity vs Developer Experience
cross = pd.crosstab(df["Developer_Experience"], df["Task_Complexity"])
cross = cross.reindex(index=["Beginner", "Intermediate", "Expert"], columns=["Low", "Medium", "High"])
plt.figure(figsize=(6.5, 5))
sns.heatmap(cross, annot=True, fmt="d", cmap="Purples", cbar_kws={"label": "Number of Tasks"})
plt.title("Figure 5.2: Crosstab \u2014 Developer Experience vs. Task Complexity")
plt.tight_layout()
plt.savefig("../images/fig5_2_crosstab_heatmap.png", dpi=150)
plt.close()

# Figure 5.3: Distributions grid (2x2)
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
sns.histplot(df["Time_Taken_Min"], bins=20, color="#4C72B0", ax=axes[0, 0])
axes[0, 0].set_title("Time Taken (min)")
sns.histplot(df["Bugs_Found"], bins=13, color="#DD8452", discrete=True, ax=axes[0, 1])
axes[0, 1].set_title("Bugs Found")
sns.histplot(df["Code_Quality_Score"], bins=15, color="#55A868", ax=axes[1, 0])
axes[1, 0].set_title("Code Quality Score")
sns.histplot(df["Satisfaction_Score"], bins=15, color="#C44E52", ax=axes[1, 1])
axes[1, 1].set_title("Satisfaction Score")
fig.suptitle("Figure 5.3: Distributions of Key Numerical Features", y=0.995)
plt.tight_layout()
plt.savefig("../images/fig5_3_distributions_grid.png", dpi=150)
plt.close()

# Figure 5.4: Correlation heatmap
corr = df[num_cols].corr()
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            cbar_kws={"label": "Correlation"})
plt.title("Figure 5.4: Correlation Heatmap of Key Numerical Features")
plt.tight_layout()
plt.savefig("../images/fig5_4_correlation_heatmap.png", dpi=150)
plt.close()

# Figure 5.5: Time taken by Task Complexity (boxplot)
plt.figure(figsize=(7.5, 5))
sns.boxplot(data=df, x="Task_Complexity", y="Time_Taken_Min",
            order=["Low", "Medium", "High"], hue="Task_Complexity",
            palette=["#B7C9F2", "#7C93D6", "#4C5FA6"], legend=False)
plt.title("Figure 5.5: Time Taken (min) by Task Complexity")
plt.xlabel("Task Complexity")
plt.ylabel("Time Taken (minutes)")
plt.tight_layout()
plt.savefig("../images/fig5_5_time_by_complexity.png", dpi=150)
plt.close()

corr_time_bugs = corr.loc["Time_Taken_Min", "Bugs_Found"]
corr_bugs_quality = corr.loc["Bugs_Found", "Code_Quality_Score"]
print("\ncorr time-bugs:", corr_time_bugs, "corr bugs-quality:", corr_bugs_quality)

# =========================================================
# MODULE 2: BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================
satisfaction = df["Satisfaction_Score"].values
n_boot = 5000
boot_means = np.array([
    np.random.choice(satisfaction, size=len(satisfaction), replace=True).mean()
    for _ in range(n_boot)
])
sample_mean = satisfaction.mean()
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"\nBootstrap: sample mean={sample_mean:.4f}, 95% CI=({ci_low:.4f}, {ci_high:.4f})")

plt.figure(figsize=(9, 5.5))
sns.histplot(boot_means, bins=40, color=PURPLE, kde=True, stat="count")
plt.axvline(sample_mean, color="black", linestyle="--", label=f"Sample mean = {sample_mean:.3f}")
plt.axvline(ci_low, color="crimson", linestyle=":", label=f"95% CI low = {ci_low:.3f}")
plt.axvline(ci_high, color="crimson", linestyle=":", label=f"95% CI high = {ci_high:.3f}")
plt.title(f"Figure 5.6: Bootstrap Sampling Distribution of Mean Satisfaction Score (n_boot={n_boot})")
plt.xlabel("Bootstrap Sample Mean Satisfaction Score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("../images/fig5_6_bootstrap_distribution.png", dpi=150)
plt.close()

# =========================================================
# MODULE 3: HYPOTHESIS TESTING (Welch's two-sample t-test)
# =========================================================
vibe_time = df.loc[df["Method"] == "Vibe Coding", "Time_Taken_Min"]
hard_time = df.loc[df["Method"] == "Hardcoding", "Time_Taken_Min"]
t_stat, p_val = stats.ttest_ind(vibe_time, hard_time, equal_var=False)
print(f"\nWelch t-test on Time_Taken_Min: t={t_stat:.4f}, p={p_val:.3e}")
print("n vibe:", len(vibe_time), "mean:", vibe_time.mean())
print("n hard:", len(hard_time), "mean:", hard_time.mean())

plt.figure(figsize=(7, 5.5))
sns.boxplot(data=df, x="Method", y="Time_Taken_Min", hue="Method", palette=PAL, legend=False)
sns.stripplot(data=df, x="Method", y="Time_Taken_Min", color="black", alpha=0.25, size=3, jitter=0.2)
plt.title("Figure 5.7: Time Taken by Method (Welch's t-test)")
plt.xlabel("Method")
plt.ylabel("Time Taken (minutes)")
plt.tight_layout()
plt.savefig("../images/fig5_7_ttest_boxplot.png", dpi=150)
plt.close()

# =========================================================
# MODULE 4: MULTIPLE LINEAR REGRESSION (with diagnostics)
# Predict Debugging_Time_Min from Bugs_Found, Lines_of_Code, Method(dummy)
# =========================================================
reg_df = df.copy()
reg_df["Method_Vibe"] = (reg_df["Method"] == "Vibe Coding").astype(int)

X_cols = ["Bugs_Found", "Lines_of_Code", "Method_Vibe"]
X = reg_df[X_cols].values.astype(float)
y = reg_df["Debugging_Time_Min"].values.astype(float)
n, k = X.shape
X_design = np.column_stack([np.ones(n), X])  # add intercept

beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
y_pred = X_design @ beta
resid = y - y_pred
ssr = np.sum(resid ** 2)
sst = np.sum((y - y.mean()) ** 2)
r2 = 1 - ssr / sst
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
sigma2 = ssr / (n - k - 1)
cov_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
se = np.sqrt(np.diag(cov_beta))
t_vals = beta / se
p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n - k - 1)

reg_table = pd.DataFrame({
    "Predictor": ["Intercept"] + X_cols,
    "Coefficient": beta.round(4),
    "Std_Error": se.round(4),
    "t": t_vals.round(3),
    "p_value": p_vals,
})
reg_table.to_csv("../results/stats_regression_table.csv", index=False)
print(f"\nRegression: R2={r2:.4f}, Adj R2={adj_r2:.4f}, N={n}")
print(reg_table)

# Figure 5.8: Residuals vs Fitted
plt.figure(figsize=(7.5, 5.5))
plt.scatter(y_pred, resid, alpha=0.6, color=PURPLE, edgecolor="white", s=40)
plt.axhline(0, color="black", linestyle="--")
plt.title("Figure 5.8: Regression Diagnostics \u2014 Residuals vs. Fitted Values")
plt.xlabel("Fitted Values (Predicted Debugging Time, min)")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("../images/fig5_8_residuals_vs_fitted.png", dpi=150)
plt.close()

# Figure 5.9: Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y, y_pred, alpha=0.6, color=TEAL, edgecolor="white", s=40)
lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
plt.plot(lims, lims, color="black", linestyle="--")
plt.title("Figure 5.9: Actual vs. Predicted Debugging Time")
plt.xlabel("Actual Debugging Time (min)")
plt.ylabel("Predicted Debugging Time (min)")
plt.tight_layout()
plt.savefig("../images/fig5_9_actual_vs_predicted.png", dpi=150)
plt.close()

# Figure 5.10: Bugs Found vs Debugging Time with polynomial fit (nonlinearity check)
plt.figure(figsize=(7.5, 5.5))
sns.regplot(data=df, x="Bugs_Found", y="Debugging_Time_Min", order=1, scatter_kws={"alpha": 0.4, "color": PURPLE}, line_kws={"color": "black", "label": "Linear fit"})
sns.regplot(data=df, x="Bugs_Found", y="Debugging_Time_Min", order=2, scatter=False, line_kws={"color": "crimson", "linestyle": "--", "label": "Degree-2 fit"})
plt.title("Figure 5.10: Debugging Time vs. Bugs Found (Linear vs. Degree-2 Fit)")
plt.xlabel("Bugs Found")
plt.ylabel("Debugging Time (minutes)")
plt.legend()
plt.tight_layout()
plt.savefig("../images/fig5_10_polynomial_check.png", dpi=150)
plt.close()

# =========================================================
# MODULE 5: GENERALIZED LINEAR MODEL (Logistic Regression)
# Predict High_Quality (Code_Quality_Score >= median) from
# Method_Vibe, Bugs_Found, Time_Taken_Min
# =========================================================
glm_df = df.copy()
glm_df["Method_Vibe"] = (glm_df["Method"] == "Vibe Coding").astype(int)
median_quality = glm_df["Code_Quality_Score"].median()
glm_df["High_Quality"] = (glm_df["Code_Quality_Score"] >= median_quality).astype(int)
print(f"\nMedian quality threshold: {median_quality}")
print(glm_df["High_Quality"].value_counts())

Xg_cols = ["Method_Vibe", "Bugs_Found", "Time_Taken_Min"]
Xg = glm_df[Xg_cols].values.astype(float)
yg = glm_df["High_Quality"].values.astype(float)
ng = Xg.shape[0]
Xg_design = np.column_stack([np.ones(ng), Xg])

# standardize continuous predictors for numerical stability (keep dummy raw)
Xg_design_std = Xg_design.copy()
means = Xg_design[:, 1:].mean(axis=0)
stds = Xg_design[:, 1:].std(axis=0)
Xg_design_std[:, 1:] = (Xg_design[:, 1:] - means) / stds

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

beta_g = np.zeros(Xg_design_std.shape[1])
for _ in range(100):
    eta = Xg_design_std @ beta_g
    p = sigmoid(eta)
    W = p * (1 - p)
    grad = Xg_design_std.T @ (yg - p)
    Hess = -(Xg_design_std * W[:, None]).T @ Xg_design_std
    step = np.linalg.solve(Hess, grad)
    beta_g = beta_g - step
    if np.max(np.abs(step)) < 1e-8:
        break

eta_final = Xg_design_std @ beta_g
p_final = sigmoid(eta_final)
W_final = p_final * (1 - p_final)
Hess_final = -(Xg_design_std * W_final[:, None]).T @ Xg_design_std
cov_g = np.linalg.inv(-Hess_final)
se_g = np.sqrt(np.diag(cov_g))
z_vals = beta_g / se_g
p_vals_g = 2 * stats.norm.sf(np.abs(z_vals))

# convert standardized coefficients back to original scale for continuous vars
# beta_orig = beta_std / std  (for continuous predictors); intercept adjust
beta_orig = beta_g.copy()
beta_orig[1:] = beta_g[1:] / stds
beta_orig[0] = beta_g[0] - np.sum(beta_g[1:] * means / stds)
se_orig = se_g.copy()
se_orig[1:] = se_g[1:] / stds
se_orig[0] = np.nan  # intercept SE on original scale not simply derived; report standardized CI for intercept only

odds_ratio = np.exp(beta_orig)
ci_low_g = np.exp(beta_orig - 1.96 * se_orig)
ci_high_g = np.exp(beta_orig + 1.96 * se_orig)

glm_table = pd.DataFrame({
    "Predictor": ["Intercept"] + Xg_cols,
    "Coefficient": beta_orig.round(4),
    "Odds_Ratio": odds_ratio.round(3),
    "CI_low": ci_low_g.round(3),
    "CI_high": ci_high_g.round(3),
    "z": z_vals.round(3),
    "p_value": p_vals_g,
})
glm_table.to_csv("../results/stats_glm_table.csv", index=False)
print("\nGLM table:\n", glm_table)

n_high_quality = glm_df["High_Quality"].sum()
pct_high_quality = 100 * n_high_quality / ng
print(f"\n{n_high_quality} of {ng} tasks ({pct_high_quality:.1f}%) are High Quality")

# Figure 5.11: Predicted probability of High Quality vs Bugs Found (others at mean, by method)
bugs_range = np.linspace(df["Bugs_Found"].min(), df["Bugs_Found"].max(), 100)
mean_time = df["Time_Taken_Min"].mean()

plt.figure(figsize=(7.5, 5.5))
for method_val, color, label in [(1, PURPLE, "Vibe Coding"), (0, TEAL, "Hardcoding")]:
    Xnew = np.column_stack([
        np.ones_like(bugs_range),
        np.full_like(bugs_range, method_val),
        bugs_range,
        np.full_like(bugs_range, mean_time),
    ])
    Xnew_std = Xnew.copy()
    Xnew_std[:, 1:] = (Xnew[:, 1:] - means) / stds
    probs = sigmoid(Xnew_std @ beta_g)
    plt.plot(bugs_range, probs, color=color, label=label, linewidth=2.5)

plt.title("Figure 5.11: Predicted Probability of High Code Quality vs. Bugs Found\n(Time Taken held at its mean)")
plt.xlabel("Bugs Found")
plt.ylabel("Predicted Probability of High Quality")
plt.legend()
plt.tight_layout()
plt.savefig("../images/fig5_11_predicted_probability.png", dpi=150)
plt.close()

print("\nAll Module 1-5 statistics and figures generated successfully.")
