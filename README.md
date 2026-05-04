# ORIE 3120 F1 Analysis Project

Formula 1 data analysis project investigating whether pole position predicts race wins (2003–2022). Includes exploratory visualizations, linear and logistic regression, residual diagnostics, and train/test predictive model comparison across eras.

## Structure

```
├── scripts/           # Python analysis and visualization scripts
├── plots/             # Generated visualization outputs (PNG)
├── Formula 1 Dataset Cleaned/   # Source F1 data (CSV)
├── Test Cases/        # Test files for validation
├── f1_merged.csv      # Merged qualifying + results dataset
└── requirements.txt   # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Merge data** (creates `f1_merged.csv`):
   ```bash
   python scripts/f1_merge.py
   ```

2. **Generate visualizations (Projetc Milestone 1)**:
   ```bash
   python scripts/viz_pole_win.py            # Pole-to-win conversion rate
   python scripts/viz_circuit_type.py         # Circuit type analysis
   python scripts/viz_grid_finish_era.py      # Grid vs finish correlations
   python scripts/viz_pitstops.py             # Pit stop analysis
   python scripts/viz_win_rate_by_grid.py     # Win rate by starting grid position
   ```

3. **Run data analysis (Projetc Milestone 2)**:
   ```bash
   python scripts/analysis_linear_regression.py    # Linear regression: grid -> finish position
   python scripts/analysis_residual_plots.py       # Residual diagnostics & assumption testing
   python scripts/analysis_logistic_regression.py   # Logistic regression: pole -> win probability
   python scripts/analysis_train_test.py            # Train/test split model comparison by era
   ```

4. **Validate**:
   ```bash
   python scripts/validate_fixes.py
   python "Test Cases/pole_win_rate_testcase.py"
   python "Test Cases/viz_circuit_type_testcase.py"
   python "Test Cases/gridvsfinish_testcase.py"
   python "Test Cases/pitspots_testcase.py"
   python "Test Cases/f1_merged_testcase.py"
   ```

## Data

Analysis uses cleaned F1 datasets (2003–2022). Pit stop analysis uses data from 2011 onward.
