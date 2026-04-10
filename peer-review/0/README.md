# STAT 214 Lab 1 - PECARN TBI Data

Exploratory data analysis and modeling of pediatric traumatic brain injury data from the PECARN study.

## Prerequisites

1. **Data**: Place `TBI PUD 10-08-2013.csv` in `lab1/data/`. Obtain from the stat-214-gsi repo.
2. **Environment**:
   ```bash
   conda env create -f code/environment.yaml
   conda activate stat214
   ```

## Reproducing the Analysis (One Command)

```bash
cd code
bash run.sh
```

This will:
1. Install Playwright Chromium (if not already present)
2. Run `models.py` to train PECARN, logistic regression, and XGBoost models
3. Execute the notebook (`report_lab1.ipynb`) end-to-end
4. Export the executed notebook to `report/report_lab1.pdf`

## Project Structure

```
lab1/
├── code/
│   ├── clean.py               # Data cleaning + reality/stability checks
│   ├── models.py              # PECARN, logistic regression, XGBoost
│   ├── run.sh                 # Full reproducible pipeline
│   ├── report_lab1.ipynb      # Main analysis notebook
│   └── environment.yaml       # Conda environment spec
├── data/                      # Place TBI PUD 10-08-2013.csv here (not tracked)
├── report/
│   ├── lab1-template.tex      # Report LaTeX source
│   └── report_lab1.pdf        # Auto-generated notebook PDF
├── instructions/
│   └── lab1-instructions.tex
└── README.md
```

## Code Style

Run `ruff check . --fix` in `code/` to check and auto-fix style issues.
