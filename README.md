# Data Preprocessing: Internet Advertisements Dataset

## Overview
This folder contains the data preprocessing pipeline (`1_preprocessing.R`) for the Probability and Statistics Project (Group 11). The script is responsible for cleaning the raw dataset downloaded from Kaggle, handling missing values, and removing outliers to prepare a clean dataset for Descriptive Statistics and Machine Learning models.

## Dependencies
Ensure you have the following R packages installed before running the script:
* `dplyr` (for data manipulation and pipelining)

You can install it via the R console:
```R
install.packages("dplyr")
Input and Output
Input: add.csv - The raw dataset downloaded from the UCI Machine Learning Repository (via Kaggle). It contains over 1500 features, missing values (denoted as ?), and categorical labels (ad. / nonad.).

Output: clean_data_for_analysis.csv - The cleaned dataset containing 2474 valid observations and exactly 4 essential columns: Height, Width, Aspect_Ratio, and IsAds (binary: 1/0).

Preprocessing Pipeline (5 Steps)
Extraction: Drops unnecessary columns (like ID and text-based URL features), retaining only the 3 main geometric features (Height, Width, Aspect_Ratio) and the target label.

Typecasting: Converts string values to numeric types. Invalid characters like ? are automatically coerced into NA (Not Available). The target label is mapped to binary (ad. -> 1, nonad. -> 0).

Imputation: * Replaces NA values in Height and Width with their respective Medians. The median is chosen over the mean because image dimensions are highly right-skewed.

Recalculates missing Aspect_Ratio mathematically (Width / Height) to maintain logical consistency.

Outlier Removal (IQR Method): Filters out extreme outliers using the Interquartile Range (IQR) method.

Critical Note: The IQR boundaries are calculated separately for the Ad group and Non-Ad group (group_by(IsAds)). This prevents wide advertisement banners from being incorrectly flagged and deleted as outliers.

Export: Generates the final .csv file.

How to Run
Open your terminal (PowerShell/Command Prompt) in this directory and execute:

Bash
& "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" 1_preprocessing.R