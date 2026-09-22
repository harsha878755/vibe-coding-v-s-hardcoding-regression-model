# Vibe Coding vs. Hardcoding — A Statistical Analysis

A mini project for comparing two software development approaches — **Vibe Coding** (AI/prompt-assisted programming) and **Hardcoding**
(traditional manual programming) — using a full statistical pipeline: EDA, bootstrap resampling,hypothesis testing, multiple regression, and a generalized linear model (logistic regression).

> The dataset is **simulated** (220 coding-task records) since no public benchmark dataset yet
> exists comparing these two approaches. Every statistic and figure in the report is computed live
> from `data/vibe_coding_vs_hardcoding.csv` — nothing is hard-coded or fabricated after the fact.

## Project Structure

```
vibe-coding-vs-hardcoding/
├── data/
│   └── vibe_coding_vs_hardcoding.csv     # the dataset (220 rows, 10 columns)
├── src/
│   └── analysis.py                        # full analysis script (run this)
├── images/
│   └── fig5_*.png                         # all generated charts (11 figures)
├── results/
│   ├── stats_overall_describe.csv         # descriptive statistics
│   ├── stats_group_means.csv              # group means by Method
│   ├── stats_regression_table.csv         # OLS regression coefficients
│   └── stats_glm_table.csv                # logistic regression odds ratios
├── report/
│   ├── Vibe_Coding_vs_Hardcoding_Report.docx
│   └── Vibe_Coding_vs_Hardcoding_Presentation.pptx
├── requirements.txt
└── README.md
```

## The Five Modules Covered

| Module | Topic | What It Answers |
|---|---|---|
| 1 | Exploring Two or More Variables | How do stress, complexity, and experience relate to outcomes? |
| 2 | The Bootstrap and Confidence Intervals | How reliable is the average satisfaction score? |
| 3 | Hypothesis Testing (Two-Sample t-Test) | Is Vibe Coding *really* faster, or just luck? |
| 4 | Factor Variables in Regression + Diagnostics | What predicts debugging time? |
| 5 | Generalized Linear Models (Odds Ratios) | What predicts whether code is rated high-quality? |

## Key Findings

- **Speed:** Vibe Coding tasks average **29.96 min** vs. **50.34 min** for Hardcoding (Welch's t-test, p < 0.00001)
- **Bugs:** Vibe Coding tasks have more bugs on average (4.44 vs. 2.67)
- **Debugging time:** driven almost entirely by bug count (R² = 0.892), not by method itself
- **Code quality:** Method is the strongest predictor of high-quality code — Hardcoding tasks are far more likely to be rated high quality (odds ratio = 0.167 for Vibe Coding, p < 0.001)

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/vibe-coding-vs-hardcoding.git
cd vibe-coding-vs-hardcoding

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis (regenerates all figures + results tables)
cd src
python analysis.py
```

## Tools Used

- **Pandas** — data loading, cleaning, cross-tabulation
- **NumPy** — bootstrap resampling, OLS regression via linear algebra
- **SciPy** — Welch's two-sample t-test
- **Matplotlib / Seaborn** — all visualizations

## Author

**Harsha M** — 1MP23AD013
Dept. of Artificial Intelligence and Data Science, BGS College of Engineering and Technology
Guide: Ms. Sindhu G, Assistant Professor

## License

This project is for academic coursework purposes.
