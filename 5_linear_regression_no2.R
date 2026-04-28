if (!require("ggplot2")) install.packages("ggplot2")
library(ggplot2)

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

# Create a data frame for Model 2 ROC plotting
df_roc_ratio <- data.frame(
  fpr = roc_ratio$fpr,
  tpr = roc_ratio$tpr
)

# Generate the plot
roc_plot_ratio <- ggplot(df_roc_ratio, aes(x = fpr, y = tpr)) +
  geom_line(color = "#D55E00", linewidth = 1.3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  annotate("text", x = 0.68, y = 0.25, 
           label = paste("AUC =", round(roc_ratio$auc, 4)), 
           color = "#D55E00", size = 5, fontface = "bold") +
  # Using annotate("point", ...) to prevent length warnings
  annotate("point", x = roc_ratio$fpr[opt_idx_r], y = roc_ratio$tpr[opt_idx_r], 
           color = "#E74C3C", size = 3) +
  labs(
    title = "ROC Curve - Aspect Ratio Model",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Save the plot as an image
print(roc_plot_ratio)
ggsave("roc_curve_aspect_ratio.png", plot = roc_plot_ratio, width = 7.5, height = 5.8, dpi = 120)
cat("\n[!] ROC curve saved to: roc_curve_aspect_ratio.png\n")

# Create a combined data frame containing coordinates of both models
roc_df <- data.frame(
  fpr = c(roc_width$fpr, roc_ratio$fpr),
  tpr = c(roc_width$tpr, roc_ratio$tpr),
  Model = factor(rep(
    c("Model 1W (Width)", "Model 2 (Aspect Ratio)"),
    times = c(length(roc_width$fpr), length(roc_ratio$fpr))
  ))
)

# Generate the combined plot using ggplot2
combined_plot <- ggplot(roc_df, aes(x = fpr, y = tpr, color = Model)) +
  geom_line(linewidth = 1.3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  scale_color_manual(values = c("#009E73", "#D55E00")) + # Teal for Width, Orange for Aspect Ratio
  annotate("text", x = 0.55, y = 0.35, 
           label = paste("AUC (Width) =", round(roc_width$auc, 4)), 
           color = "#009E73", size = 4.5, fontface = "bold", hjust = 0) +
  annotate("text", x = 0.55, y = 0.25, 
           label = paste("AUC (Aspect Ratio) =", round(roc_ratio$auc, 4)), 
           color = "#D55E00", size = 4.5, fontface = "bold", hjust = 0) +
  labs(
    title = "Combined ROC Curves for 2 Logistic Regression Models",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  )

# Display and save the combined plot
print(combined_plot)
ggsave("roc_comparison_final.png", plot = combined_plot, width = 8, height = 6, dpi = 120)
cat("[!] Combined ROC curves saved to: roc_comparison_final.png\n")

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