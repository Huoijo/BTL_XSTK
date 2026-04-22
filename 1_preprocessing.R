library(dplyr)

# 1. READ DATA
cat("Reading data...\n")
# Assuming the file downloaded from Kaggle is named "add.csv"
# header = FALSE because the dataset lacks column names
# stringsAsFactors = FALSE prevents R from converting text to factors early
raw_data <- read.csv("add.csv", header = FALSE, stringsAsFactors = FALSE)

# 2. EXTRACT COLUMNS AND RENAME
# Drop ID (col 1), keep Height (2), Width (3), Aspect_Ratio (4), Target (ncol)
process_data <- raw_data[, c(2, 3, 4, ncol(raw_data))]
colnames(process_data) <- c("Height", "Width", "Aspect_Ratio", "IsAds")

# 3. TYPECASTING & TARGET ENCODING
cat("Typecasting and encoding target variable...\n")
process_data <- process_data |>
  mutate(
    # trimws() removes spaces; as.numeric() converts chars to numbers
    Height = as.numeric(trimws(Height)),
    Width = as.numeric(trimws(Width)),
    Aspect_Ratio = as.numeric(trimws(Aspect_Ratio)),
    # Encode target label: "ad." becomes 1, "nonad." becomes 0
    IsAds = ifelse(trimws(IsAds) == "ad.", 1, 0)
  )

# 4. MISSING DATA IMPUTATION
cat("Imputing missing values...\n")
process_data <- process_data |>
  mutate(
    # Replace NAs in Height and Width with their respective median values
    Height = ifelse(is.na(Height), median(Height, na.rm = TRUE), Height),
    Width = ifelse(is.na(Width), median(Width, na.rm = TRUE), Width),
    # Recalculate missing Aspect_Ratio mathematically (Width / Height)
    Aspect_Ratio = ifelse(is.na(Aspect_Ratio), Width / Height, Aspect_Ratio)
  )

# 5. OUTLIER REMOVAL USING IQR METHOD
cat("Removing outliers grouped by target class...\n")
clean_data <- process_data |>
  # CRITICAL: Group by IsAds to calculate IQR for Ads and Non-Ads separately
  group_by(IsAds) |>
  mutate(
    h_Q1 = quantile(Height, 0.25),
    h_Q3 = quantile(Height, 0.75),
    h_IQR = h_Q3 - h_Q1,
    h_lower = h_Q1 - 1.5 * h_IQR,
    h_upper = h_Q3 + 1.5 * h_IQR,

    w_Q1 = quantile(Width, 0.25),
    w_Q3 = quantile(Width, 0.75),
    w_IQR = w_Q3 - w_Q1,
    w_lower = w_Q1 - 1.5 * w_IQR,
    w_upper = w_Q3 + 1.5 * w_IQR
  ) |>
  # Keep only the rows within the safe boundaries
  filter(
    Height >= h_lower & Height <= h_upper & Width >= w_lower & Width <= w_upper
  ) |>
  # Clean up temporary boundary columns
  select(-starts_with("h_"), -starts_with("w_")) |>
  ungroup()

# 6. PRINT SUMMARY AND EXPORT CLEAN DATA
cat("=========================================\n")
cat("Initial row count:", nrow(raw_data), "rows\n")
cat("Row count after outlier removal:", nrow(clean_data), "rows\n")
cat("Total outliers removed:", nrow(raw_data) - nrow(clean_data), "rows\n")
cat("=========================================\n")

# Export to a new CSV file
write.csv(clean_data, "clean_data_for_analysis.csv", row.names = FALSE)
cat("Successfully exported: clean_data_for_analysis.csv\n")