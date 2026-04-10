# Lab 1 - PECARN TBI Data
**STAT 214, Spring 2026**

## Overview
Analysis of the PECARN pediatric head trauma dataset, including data cleaning, exploratory analysis, and predictive modeling for clinically important traumatic brain injury (ciTBI).

## Setup
```bash
conda env create -f code/environment.yaml
conda activate stat214
```

## Reproducing the Analysis
From the `lab1/` directory:
```bash
bash code/run.sh
```

## Repository Structure

The general tree structure of my repo is below. All required files and directories (`code/`, `data/`, `report/`) match the expected structure. The `documents/` directory contains additional materials not required by the assignment: `dslc_documentation/` holds my working Jupyter notebooks and the `new_col_names.csv` mapping file I built to rename the raw columns to human-readable names. Figures are split across two subdirectories — `other_figures/` contains exploratory figures generated during the notebook analysis (some of which appear in the final report), and `main_figures/` contains the publication-quality figures for the three main findings.

Note the data subdirectory is only on my local machine as follows, because as instructed .gitignore prevents it from being pushed to my online repo.

```
lab1/
├── code/
│   ├── clean.py
│   ├── models.py
│   ├── run.sh
│   └── environment.yaml
├── data/
│   ├── TBI PUD 10-08-2013.csv
│   ├── TBI_PUD_cleaned.csv
│   └── TBI_PUD_cleaned_perturbed.csv
├── documents/
│   ├── dslc_documentation/
│   │   ├── 01_cleaning.ipynb
│   │   ├── 02_eda.ipynb
│   │   ├── 03_stability_and_modeling.ipynb
│   │   └── new_col_names.csv
|   |   └── other_figures/
|   └── main_figures/
│   └── Kuppermann_2009_The-Lancet_000.pdf
├── instructions/
|   └── lab1-instructions.pdf 
|   └── lab1-instructions.tex
|   └── lab1-template.pdf
|   └── lab1-template.tex
├── report/
│   └── lab1.tex
    └── lab1.pdf
└── README.md
```