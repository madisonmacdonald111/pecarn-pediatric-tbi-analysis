#!/usr/bin/env bash

# Get the directory of this script and cd into it
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

echo "Running Lab 1 Pipeline..."

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stat214

# Step 0: Ensure Playwright Chromium is installed (no-op if already present)
echo "Ensuring Playwright Chromium is installed..."
playwright install chromium

# Step 1: Run clean.py
echo "Running clean.py..."
python clean.py --data-path "../data/TBI PUD 10-08-2013.csv"

# Step 2: Run models.py
echo "Running models.py..."
python models.py --data-path "../data/TBI PUD 10-08-2013.csv"

# Step 3: Execute the notebook to generate all figures and results
echo "Executing notebook..."
jupyter nbconvert --to notebook --execute --inplace report_lab1.ipynb

# Step 4: Export the executed notebook to PDF (via webpdf/Playwright)
echo "Exporting notebook to PDF..."
jupyter nbconvert --to webpdf --no-input \
    report_lab1.ipynb \
    --output-dir "../report" \
    --output "report_lab1"
echo "PDF saved to report/report_lab1.pdf"

# Deactivate conda environment
conda deactivate
