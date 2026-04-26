# Ad Image Classification with Logistic Regression

## What This Code Does
Predicts whether an image is an advertisement based on its height, width, and aspect ratio using logistic regression.

## Requirements
- **R** (any version 3.6+)
- **Input file**: `clean_data_for_analysis.csv` with columns:
  - `Height` (numeric)
  - `Width` (numeric)  
  - `Aspect_Ratio` (numeric)
  - `IsAds` (0 = non-ad, 1 = ad)

## How to Run
```bash
Rscript logistic_regression_model.R