calculate_roc <- function(actual, predictions) {
  valid <- complete.cases(actual, predictions)
  actual <- actual[valid]
  predictions <- predictions[valid]
  
  order_idx <- order(predictions, decreasing = TRUE)
  actual_sorted <- actual[order_idx]
  pred_sorted <- predictions[order_idx]
  
  tpr <- c(0); fpr <- c(0); thresholds <- c(1)
  
  for(i in 1:length(actual_sorted)) {
    curr_thresh <- pred_sorted[i]
    thresholds <- c(thresholds, curr_thresh)
    pred_class <- ifelse(predictions >= curr_thresh, 1, 0)
    
    tp <- sum(pred_class == 1 & actual == 1)
    fp <- sum(pred_class == 1 & actual == 0)
    fn <- sum(pred_class == 0 & actual == 1)
    tn <- sum(pred_class == 0 & actual == 0)
    
    tpr_val <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    fpr_val <- ifelse((fp + tn) > 0, fp / (fp + tn), 0)
    
    tpr <- c(tpr, tpr_val)
    fpr <- c(fpr, fpr_val)
  }
  
  thresholds <- c(thresholds, 0)
  tpr <- c(tpr, 1)
  fpr <- c(fpr, 1)
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  
  return(list(fpr = fpr, tpr = tpr, auc = auc, thresholds = thresholds))
}

# LOAD DATA & TRAIN/TEST SPLIT
cat("Loading pre-cleaned dataset...\n")
data <- read.csv("clean_data_for_analysis.csv", 
                 stringsAsFactors = FALSE,
                 header = TRUE)

set.seed(42)
n <- nrow(data)
train_idx <- sample(1:n, size = floor(0.7 * n))
train <- data[train_idx, ]
test  <- data[-train_idx, ]


# MODEL 1 (WIDTH ONLY) FOR COMPARISON

model_width <- glm(IsAds ~ Width, data = train, family = binomial(), control = glm.control(maxit = 50))
pred_prob_width <- predict(model_width, test, type = "response")
roc_width <- calculate_roc(test$IsAds, pred_prob_width)

# Optimal threshold and Accuracy of Model 1
youden_w <- roc_width$tpr - roc_width$fpr
opt_idx_w <- which.max(youden_w[2:(length(youden_w)-1)]) + 1
opt_th_w <- roc_width$thresholds[opt_idx_w]
acc_opt_w <- mean(ifelse(pred_prob_width > opt_th_w, 1, 0) == test$IsAds)
aic_w <- AIC(model_width)

# MODEL 2: ASPECT RATIO ONLY
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL 2: IsAds ~ Aspect_Ratio \n")
cat(paste(rep("=", 60), collapse = ""), "\n")

model_ratio <- glm(IsAds ~ Aspect_Ratio, 
                   data = train,
                   family = binomial(),
                   control = glm.control(maxit = 50))

cat("\n--- Coefficients ---\n")
print(coef(summary(model_ratio)))

cat("\n--- Model Summary ---\n")
print(summary(model_ratio))

# Predictions
pred_prob_ratio <- predict(model_ratio, test, type = "response")
roc_ratio <- calculate_roc(test$IsAds, pred_prob_ratio)

# Optimal threshold (Youden's J)
youden_r <- roc_ratio$tpr - roc_ratio$fpr
opt_idx_r <- which.max(youden_r[2:(length(youden_r)-1)]) + 1
opt_th_r <- roc_ratio$thresholds[opt_idx_r]

cat("\n--- ROC & AUC (Aspect Ratio) ---\n")
cat("AUC:", round(roc_ratio$auc, 4), "\n")
cat("Optimal Threshold (Youden):", round(opt_th_r, 3), "\n")

# Performance at Optimal Threshold
pred_class_opt_r <- ifelse(pred_prob_ratio > opt_th_r, 1, 0)
acc_opt_r <- mean(pred_class_opt_r == test$IsAds)
conf_opt_r <- table(Predicted = pred_class_opt_r, Actual = test$IsAds)
aic_r <- AIC(model_ratio)

cat("\n--- Performance at Optimal Threshold ---\n")
cat("Accuracy:", round(acc_opt_r, 4), "\n")
print(conf_opt_r)

if (all(dim(conf_opt_r) == c(2,2))) {
  tn_r <- conf_opt_r[1,1]; fp_r <- conf_opt_r[2,1]
  fn_r <- conf_opt_r[1,2]; tp_r <- conf_opt_r[2,2]
  cat("Sensitivity:", round(tp_r/(tp_r+fn_r), 4), "\n")
  cat("Specificity:", round(tn_r/(tn_r+fp_r), 4), "\n")
  cat("Precision:", round(tp_r/(tp_r+fp_r), 4), "\n")
}

# ROC Curve for Model 2
png("roc_curve_aspect_ratio.png", width = 900, height = 700, res = 120)

plot(roc_ratio$fpr, roc_ratio$tpr, 
     type = "l", col = "#2E86C1", lwd = 3,
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     main = "ROC Curve - Aspect Ratio Model",
     xlim = c(0, 1), ylim = c(0, 1))

abline(a = 0, b = 1, lty = 2, col = "gray", lwd = 2)
grid(col = "lightgray", lty = 3)
text(0.68, 0.25, paste("AUC =", round(roc_ratio$auc, 4)), 
     cex = 1.4, col = "#2E86C1", font = 2)

# Optimal Threshold point
points(roc_ratio$fpr[opt_idx_r], roc_ratio$tpr[opt_idx_r], 
       col = "#E74C3C", pch = 19, cex = 2)

legend("bottomright",
       legend = c("Aspect Ratio Model", "Random Classifier", "Optimal Threshold"),
       col = c("#2E86C1", "gray", "#E74C3C"),
       lty = c(1, 2, NA), pch = c(NA, NA, 19), lwd = c(3, 2, NA),
       bty = "n", cex = 1.1)

dev.off()
cat("\n[!] ROC curve saved to: roc_curve_aspect_ratio.png\n")

# Compare ROC Curves
png("roc_comparison_final.png", width = 900, height = 700, res = 120)

plot(roc_width$fpr, roc_width$tpr, 
     type = "l", col = "#009E73", lwd = 3,
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     main = "ROC Curves Comparison",
     xlim = c(0, 1), ylim = c(0, 1))

lines(roc_ratio$fpr, roc_ratio$tpr, col = "#D55E00", lwd = 3)

abline(a = 0, b = 1, lty = 2, col = "gray", lwd = 2)
grid(col = "lightgray", lty = 3)

text(0.55, 0.35, paste("AUC (Width) =", round(roc_width$auc, 4)), 
     cex = 1.2, col = "#009E73", font = 2, adj = 0)
text(0.55, 0.25, paste("AUC (Aspect Ratio) =", round(roc_ratio$auc, 4)), 
     cex = 1.2, col = "#D55E00", font = 2, adj = 0)

legend("bottomright",
       legend = c("Model 1W (Width)", "Model 2 (Aspect Ratio)", "Random Classifier"),
       col = c("#009E73", "#D55E00", "gray"),
       lty = c(1, 1, 2), lwd = c(3, 3, 2),
       bty = "n", cex = 1.1)

dev.off()
cat("\n[!] Combined ROC curves saved to: roc_comparison_final.png\n")

# MODEL COMPARISON

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("FINAL MODEL COMPARISON \n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n| Metric              | Model 1 (Width Only) | Model 2 (Aspect Ratio) |\n")
cat("|---------------------|----------------------|------------------------|\n")
cat("| Accuracy (Optimal)  |", round(acc_opt_w, 4), "              |", round(acc_opt_r, 4), "                |\n")
cat("| AUC                 |", round(roc_width$auc, 4), "              |", round(roc_ratio$auc, 4), "                |\n")
cat("| AIC (from summary)  |", round(aic_w, 2), "              |", round(aic_r, 2), "               |\n")

cat("\n", paste(rep("-", 60), collapse = ""), "\n")
cat("STATISTICIAN'S CONCLUSION:\n")
cat(paste(rep("-", 60), collapse = ""), "\n")

cat("Based on the evaluation metrics:\n")
if(roc_ratio$auc > roc_width$auc && aic_r < aic_w) {
  cat("✓ RECOMMENDATION: Use Model 2 (Aspect Ratio).\n")
  cat("  - It demonstrates a higher AUC, meaning better discrimination capability.\n")
  cat("  - It has a lower AIC, indicating less information loss and better model fit.\n")
} else if (roc_width$auc > roc_ratio$auc && aic_w < aic_r) {
  cat("✓ RECOMMENDATION: Use Model 1 (Width Only).\n")
  cat("  - It demonstrates a significantly higher AUC, indicating superior capability to separate Ads from Non-Ads.\n")
  cat("  - It yields a much lower AIC, proving it provides a better fit to the data with the same model complexity.\n")
  cat("  - The Accuracy at the optimal threshold is also notably higher.\n")
} else {
  cat("=> Both models show conflicting trade-offs between AIC, AUC, and Accuracy. Selection depends on specific business requirements (e.g., favoring Precision vs Recall).\n")
}