#!/bin/bash
conda activate stat214

python code/clean.py "data/TBI PUD 10-08-2013.csv"

python code/clean.py "data/TBI PUD 10-08-2013.csv" 13    # to get the perturbed data for stability check 

python code/models.py data/TBI_PUD_cleaned.csv 

python code/models.py data/TBI_PUD_cleaned_perturbed.csv    # use perturbed data for stability check 

conda deactivate