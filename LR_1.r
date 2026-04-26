# ============================================================================
# LOGISTIC REGRESSION FOR AD IMAGE CLASSIFICATION
# ============================================================================

#1. LOAD DATA
cat("Loading pre-cleaned dataset...\n")
data <- read.csv("clean_data_for_analysis.csv", 
                 stringsAsFactors = FALSE,    # Keep strings as characters, not factors
                 header = TRUE)               # First row contains column names

# Quick check to verify data loaded correctly
cat("Dataset dimensions:", dim(data), "\n")      # Shows rows and columns
cat("Column names:", colnames(data), "\n")       # Lists all feature names
cat("First few rows:\n")
print(head(data))                                 # Preview first 6 rows

# 2. TRAIN/TEST SPLIT (70/30 split)
set.seed(42)  # Set random seed for reproducibility (same split every run)
n <- nrow(data)  # Total number of observations

# Randomly sample 70% of row indices without replacement
train_idx <- sample(1:n, size = floor(0.7 * n))

# Create training set (70% of data) and test set (remaining 30%)
train <- data[train_idx, ]   # Rows for training
test  <- data[-train_idx, ]  # Rows for testing

cat("Training set size:", nrow(train), "\n")
cat("Test set size:", nrow(test), "\n")

# 3. TRAIN LOGISTIC REGRESSION MODEL
cat("\nTraining logistic regression model...\n")
model <- glm(IsAds ~ .,           # Target ~ all features
             data = train,
             family = binomial(), # Specifies logistic regression
             control = glm.control(maxit = 50))  # Allow up to 50 iterations

# Display model coefficients with statistical significance
cat("\n=========================================\n")
cat("MODEL COEFFICIENTS:\n")
cat("=========================================\n")
print(coef(summary(model)))

# 4. MAKE PREDICTIONS ON TEST SET
pred_prob <- predict(model, test, type = "response")  # Probability predictions
pred_class <- ifelse(pred_prob > 0.5, 1, 0)           # Convert to binary classes

# 5. CALCULATE AND DISPLAY ACCURACY
accuracy <- mean(pred_class == test$IsAds)  # Proportion of correct predictions
cat("\n=========================================\n")
cat("TEST SET ACCURACY:", round(accuracy, 4), "\n")
cat("=========================================\n")

# 6. CONFUSION MATRIX
conf_matrix <- table(Predicted = pred_class, Actual = test$IsAds)
cat("\nConfusion Matrix:\n")
print(conf_matrix)

# 7. ADDITIONAL PERFORMANCE METRICS
if (all(dim(conf_matrix) == c(2,2))) {
  tn <- conf_matrix[1,1]
  fp <- conf_matrix[2,1]
  fn <- conf_matrix[1,2]
  tp <- conf_matrix[2,2]
  
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  precision <- tp / (tp + fp)
  
  cat("\n=========================================\n")
  cat("PERFORMANCE METRICS (with threshold = 0.5):\n")
  cat("=========================================\n")
  cat("Sensitivity (True Positive Rate):", round(sensitivity, 4), "\n")
  cat("   Proportion of actual ads correctly identified\n")
  cat("Specificity (True Negative Rate):", round(specificity, 4), "\n")
  cat("   Proportion of actual non-ads correctly identified\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("   Probability that a predicted ad is actually an ad\n")
}

# ============================================================================
# 8. ROC CURVE (No Packages Required)
# ============================================================================
cat("\n=========================================\n")
cat("GENERATING ROC CURVE:\n")
cat("=========================================\n")

# Calculate ROC manually
calculate_roc <- function(actual, predictions) {
  # Handle potential NAs
  valid <- complete.cases(actual, predictions)
  actual <- actual[valid]
  predictions <- predictions[valid]
  
  # Sort by predicted probability
  order_idx <- order(predictions, decreasing = TRUE)
  actual_sorted <- actual[order_idx]
  pred_sorted <- predictions[order_idx]
  
  # Add start point
  tpr <- c(0)
  fpr <- c(0)
  thresholds <- c(1)
  
  # Calculate at each unique prediction
  for(i in 1:length(actual_sorted)) {
    curr_thresh <- pred_sorted[i]
    thresholds <- c(thresholds, curr_thresh)
    
    # Make predictions
    pred_class <- ifelse(predictions >= curr_thresh, 1, 0)
    
    # Calculate metrics
    tp <- sum(pred_class == 1 & actual == 1)
    fp <- sum(pred_class == 1 & actual == 0)
    fn <- sum(pred_class == 0 & actual == 1)
    tn <- sum(pred_class == 0 & actual == 0)
    
    # Avoid division by zero
    tpr_val <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    fpr_val <- ifelse((fp + tn) > 0, fp / (fp + tn), 0)
    
    tpr <- c(tpr, tpr_val)
    fpr <- c(fpr, fpr_val)
  }
  
  # Add end point
  thresholds <- c(thresholds, 0)
  tpr <- c(tpr, 1)
  fpr <- c(fpr, 1)
  
  # Calculate AUC
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  
  return(list(fpr = fpr, tpr = tpr, auc = auc, thresholds = thresholds))
}

# Run ROC calculation
roc_results <- calculate_roc(test$IsAds, pred_prob)

# Save ROC plot as PNG file
png("roc_curve.png", width = 900, height = 700, res = 120, pointsize = 14)

# Create plot
plot(roc_results$fpr, roc_results$tpr, 
     type = "l", 
     col = "#2C3E50", 
     lwd = 3,
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     main = "ROC Curve - Ad Detection Model",
     xlim = c(0, 1),
     ylim = c(0, 1),
     cex.lab = 1.2,
     cex.main = 1.4,
     cex.axis = 1.1)

# Add diagonal line (random classifier)
abline(a = 0, b = 1, lty = 2, col = "gray", lwd = 2)

# Add grid
grid(col = "lightgray", lty = 3, lwd = 1)

# Add AUC text
text(0.68, 0.25, 
     paste("AUC =", round(roc_results$auc, 4)), 
     cex = 1.4,
     col = "#2C3E50",
     font = 2)

# Find and mark optimal threshold
youden <- roc_results$tpr - roc_results$fpr
optimal_idx <- which.max(youden[2:(length(youden)-1)]) + 1
optimal_threshold <- roc_results$thresholds[optimal_idx]

# Add point for optimal threshold
points(roc_results$fpr[optimal_idx], roc_results$tpr[optimal_idx], 
       col = "#E74C3C", 
       pch = 19, 
       cex = 2)

# Add label for optimal threshold
text(roc_results$fpr[optimal_idx] + 0.08, 
     roc_results$tpr[optimal_idx] - 0.05,
     paste0("Threshold = ", round(optimal_threshold, 3)),
     col = "#E74C3C",
     cex = 0.9,
     font = 2)

# Add legend
legend("bottomright",
       legend = c("Model ROC", "Random Classifier", "Optimal Threshold"),
       col = c("#2C3E50", "gray", "#E74C3C"),
       lty = c(1, 2, NA),
       pch = c(NA, NA, 19),
       lwd = c(3, 2, NA),
       pt.cex = 1.5,
       bty = "n",
       cex = 1.1)

# Close PNG device
dev.off()

cat("\n ROC curve saved to: roc_curve.png\n")
cat(" Location:", file.path(getwd(), "roc_curve.png"), "\n")

# Open the plot automatically on Mac
if (Sys.info()["sysname"] == "Darwin") {
  system("open roc_curve.png")
  cat(" Plot opened in Preview.app\n")
}

# ============================================================================
# 9. ROC CURVE SUMMARY
# ============================================================================
cat("\n=========================================\n")
cat("ROC CURVE SUMMARY:\n")
cat("=========================================\n")
cat("AUC (Area Under Curve):", round(roc_results$auc, 4), "\n")
cat("\nInterpretation:\n")
if(roc_results$auc >= 0.9) {
  cat("   Excellent discrimination (AUC > 0.9)\n")
} else if(roc_results$auc >= 0.8) {
  cat("   Good discrimination (AUC > 0.8)\n")
} else if(roc_results$auc >= 0.7) {
  cat("   Fair discrimination (AUC > 0.7)\n")
} else {
  cat("   Poor discrimination (AUC < 0.7)\n")
}

cat("\nOptimal threshold (from ROC):", round(optimal_threshold, 3), "\n")
cat("   At this threshold:\n")
cat("     • Sensitivity:", round(roc_results$tpr[optimal_idx], 4), "\n")
cat("     • Specificity:", round(1 - roc_results$fpr[optimal_idx], 4), "\n")

# ============================================================================
# 10. COMPARE DEFAULT VS OPTIMAL THRESHOLD (Optional)
# ============================================================================
cat("\n=========================================\n")
cat("THRESHOLD COMPARISON:\n")
cat("=========================================\n")

# Make predictions with optimal threshold
pred_class_optimal <- ifelse(pred_prob > optimal_threshold, 1, 0)

# Calculate metrics for optimal threshold
accuracy_optimal <- mean(pred_class_optimal == test$IsAds)

# Confusion matrix for optimal threshold
conf_matrix_optimal <- table(Predicted = pred_class_optimal, Actual = test$IsAds)

if (all(dim(conf_matrix_optimal) == c(2,2))) {
  tn_opt <- conf_matrix_optimal[1,1]
  fp_opt <- conf_matrix_optimal[2,1]
  fn_opt <- conf_matrix_optimal[1,2]
  tp_opt <- conf_matrix_optimal[2,2]
  
  sensitivity_opt <- tp_opt / (tp_opt + fn_opt)
  specificity_opt <- tn_opt / (tn_opt + fp_opt)
  precision_opt <- tp_opt / (tp_opt + fp_opt)
  
  cat("\nComparison (Default 0.5 vs Optimal", round(optimal_threshold, 3), "):\n")
  cat("  Accuracy:     ", round(accuracy, 4), " vs ", round(accuracy_optimal, 4), "\n")
  cat("  Sensitivity:  ", round(sensitivity, 4), " vs ", round(sensitivity_opt, 4), "\n")
  cat("  Specificity:  ", round(specificity, 4), " vs ", round(specificity_opt, 4), "\n")
  cat("  Precision:    ", round(precision, 4), " vs ", round(precision_opt, 4), "\n")
  
  if(accuracy_optimal > accuracy) {
    cat("\n Optimal threshold improves accuracy by", 
        round((accuracy_optimal - accuracy) * 100, 2), "%\n")
    cat("  Consider using: pred_class <- ifelse(pred_prob >", 
        round(optimal_threshold, 3), ", 1, 0)\n")
  } else {
    cat("\n Default threshold (0.5) performs better for your data\n")
  }
}

cat("\n=========================================\n")
cat("FILES CREATED:\n")
cat("  • roc_curve.png - ROC curve image\n")
cat("=========================================\n")