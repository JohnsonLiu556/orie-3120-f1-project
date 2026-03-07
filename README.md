# ORIE 3120 F1 Analysis Project

Formula 1 data analysis project covering pole-to-win conversion rates, circuit types, grid vs. finish correlations, and pit stop analysis (2003–2022).

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

2. **Generate visualizations**:
   ```bash
   python scripts/viz_pole_win.py      # Pole-to-win conversion rate
   python scripts/viz_circuit_type.py  # Circuit type analysis
   python scripts/viz_grid_finish_era.py # Grid vs finish by era
   python scripts/viz_pitstops.py      # Pit stop analysis
   ```

3. **Validate**:
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
