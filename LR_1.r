# ============================================================================
# LOGISTIC REGRESSION FOR AD IMAGE CLASSIFICATION
# ============================================================================
# SO SÁNH HAI MÔ HÌNH: (Height + Width) vs (Width Only)
# ============================================================================

# 1. LOAD DATA
cat("Loading pre-cleaned dataset...\n")
data <- read.csv("clean_data_for_analysis.csv", 
                 stringsAsFactors = FALSE,
                 header = TRUE)

cat("Dataset dimensions:", dim(data), "\n")
cat("Column names:", colnames(data), "\n")
cat("First few rows:\n")
print(head(data))

# 2. TRAIN/TEST SPLIT (70/30)
set.seed(42)
n <- nrow(data)
train_idx <- sample(1:n, size = floor(0.7 * n))
train <- data[train_idx, ]
test  <- data[-train_idx, ]

cat("\nTraining set size:", nrow(train), "\n")
cat("Test set size:", nrow(test), "\n")

# ============================================================================
# MODEL A: Height + Width (Full Model)
# ============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL A: IsAds ~ Height + Width\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

model_full <- glm(IsAds ~ Height + Width, 
                  data = train,
                  family = binomial(),
                  control = glm.control(maxit = 50))

cat("\n--- Coefficients ---\n")
print(coef(summary(model_full)))

cat("\n--- Model Summary ---\n")
print(summary(model_full))

# Predictions for Full Model
pred_prob_full <- predict(model_full, test, type = "response")
pred_class_full <- ifelse(pred_prob_full > 0.5, 1, 0)

# Performance for Full Model
accuracy_full <- mean(pred_class_full == test$IsAds)
conf_matrix_full <- table(Predicted = pred_class_full, Actual = test$IsAds)

cat("\n--- Performance (threshold = 0.5) ---\n")
cat("Accuracy:", round(accuracy_full, 4), "\n")
print(conf_matrix_full)

if (all(dim(conf_matrix_full) == c(2,2))) {
  tn <- conf_matrix_full[1,1]; fp <- conf_matrix_full[2,1]
  fn <- conf_matrix_full[1,2]; tp <- conf_matrix_full[2,2]
  cat("Sensitivity:", round(tp/(tp+fn), 4), "\n")
  cat("Specificity:", round(tn/(tn+fp), 4), "\n")
  cat("Precision:", round(tp/(tp+fp), 4), "\n")
}

# ============================================================================
# MODEL B: Width Only (Reduced Model)
# ============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL B: IsAds ~ Width Only\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

model_width <- glm(IsAds ~ Width, 
                   data = train,
                   family = binomial(),
                   control = glm.control(maxit = 50))

cat("\n--- Coefficients ---\n")
print(coef(summary(model_width)))

cat("\n--- Model Summary ---\n")
print(summary(model_width))

# Predictions for Width Model
pred_prob_width <- predict(model_width, test, type = "response")
pred_class_width <- ifelse(pred_prob_width > 0.5, 1, 0)

# Performance for Width Model
accuracy_width <- mean(pred_class_width == test$IsAds)
conf_matrix_width <- table(Predicted = pred_class_width, Actual = test$IsAds)

cat("\n--- Performance (threshold = 0.5) ---\n")
cat("Accuracy:", round(accuracy_width, 4), "\n")
print(conf_matrix_width)

if (all(dim(conf_matrix_width) == c(2,2))) {
  tn <- conf_matrix_width[1,1]; fp <- conf_matrix_width[2,1]
  fn <- conf_matrix_width[1,2]; tp <- conf_matrix_width[2,2]
  cat("Sensitivity:", round(tp/(tp+fn), 4), "\n")
  cat("Specificity:", round(tn/(tn+fp), 4), "\n")
  cat("Precision:", round(tp/(tp+fp), 4), "\n")
}

# ============================================================================
# SO SÁNH HAI MÔ HÌNH
# ============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL COMPARISON\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n| Metric              | Full Model (H+W) | Width Only |\n")
cat("|---------------------|------------------|------------|\n")
cat("| Accuracy            |", round(accuracy_full, 4), "            |", round(accuracy_width, 4), "        |\n")
cat("| AIC (from summary)  | 660.84           | 660.36     |\n")

# ============================================================================
# ROC CURVE CHO Ý KIẾN CỦA BẠN
# ============================================================================

# CÂU HỎI: Có cần vẽ ROC cho Width Only không?

# Câu trả lời: CÓ HOẶC KHÔNG tùy mục đích

# Lý do NÊN vẽ ROC cho Width Only:
# 1. Để so sánh AUC giữa hai mô hình
# 2. Để xem liệu bỏ Height có làm giảm chất lượng phân loại không
# 3. Để chứng minh Width Only model không thua kém Full model

# Lý do KHÔNG CẦN vẽ:
# 1. Nếu bạn chỉ muốn chứng minh Width là đủ
# 2. Tiết kiệm thời gian code

# Tùy chọn 1: Vẽ ROC cho Width Only (NÊN làm cho báo cáo hoàn chỉnh)

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("ROC CURVE - WIDTH ONLY MODEL\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

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

roc_width <- calculate_roc(test$IsAds, pred_prob_width)

png("roc_curve_width_only.png", width = 900, height = 700, res = 120)

plot(roc_width$fpr, roc_width$tpr, 
     type = "l", col = "#009E73", lwd = 3,
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     main = "ROC Curve - Width Only Model",
     xlim = c(0, 1), ylim = c(0, 1))

abline(a = 0, b = 1, lty = 2, col = "gray", lwd = 2)
grid(col = "lightgray", lty = 3)
text(0.68, 0.25, paste("AUC =", round(roc_width$auc, 4)), 
     cex = 1.4, col = "#009E73", font = 2)

youden <- roc_width$tpr - roc_width$fpr
optimal_idx <- which.max(youden[2:(length(youden)-1)]) + 1
optimal_threshold <- roc_width$thresholds[optimal_idx]
# ============================================================================
# ĐÁNH GIÁ TẠI NGƯỠNG TỐI ƯU (YOU DEN)
# ============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("PERFORMANCE AT OPTIMAL THRESHOLD (Width Only)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Dùng optimal_threshold vừa tính được
pred_class_opt <- ifelse(pred_prob_width > optimal_threshold, 1, 0)

# Accuracy
accuracy_opt <- mean(pred_class_opt == test$IsAds)

# Confusion matrix
conf_opt <- table(Predicted = pred_class_opt, Actual = test$IsAds)
cat("\nConfusion Matrix (optimal threshold =", round(optimal_threshold, 3), "):\n")
print(conf_opt)

if (all(dim(conf_opt) == c(2,2))) {
  tn_opt <- conf_opt[1,1]; fp_opt <- conf_opt[2,1]
  fn_opt <- conf_opt[1,2]; tp_opt <- conf_opt[2,2]
  
  sensitivity_opt <- tp_opt / (tp_opt + fn_opt)
  specificity_opt <- tn_opt / (tn_opt + fp_opt)
  precision_opt <- tp_opt / (tp_opt + fp_opt)
  
  cat("\nPerformance at optimal threshold:\n")
  cat("  Accuracy   :", round(accuracy_opt, 4), "\n")
  cat("  Sensitivity:", round(sensitivity_opt, 4), "\n")
  cat("  Specificity:", round(specificity_opt, 4), "\n")
  cat("  Precision  :", round(precision_opt, 4), "\n")
  
  # Youden's J (xác nhận)
  youden_opt <- sensitivity_opt + specificity_opt - 1
  cat("  Youden's J :", round(youden_opt, 4), "\n")
}

# So sánh với ngưỡng 0.5 (đã có sẵn)
cat("\nComparison vs default threshold (0.5):\n")
cat("  Accuracy:   0.5 =", round(accuracy_width, 4), " | optimal =", round(accuracy_opt, 4), "\n")
cat("  Sensitivity:0.5 =", round(tp/(tp+fn), 4), " | optimal =", round(sensitivity_opt, 4), "\n")
cat("  Specificity:0.5 =", round(tn/(tn+fp), 4), " | optimal =", round(specificity_opt, 4), "\n")
points(roc_width$fpr[optimal_idx], roc_width$tpr[optimal_idx], 
       col = "#E74C3C", pch = 19, cex = 2)

legend("bottomright",
       legend = c("Width Only Model", "Random Classifier", "Optimal Threshold"),
       col = c("#009E73", "gray", "#E74C3C"),
       lty = c(1, 2, NA), pch = c(NA, NA, 19), lwd = c(3, 2, NA),
       bty = "n", cex = 1.1)

dev.off()
cat("ROC curve saved to: roc_curve_width_only.png\n")

if (Sys.info()["sysname"] == "Darwin") system("open roc_curve_width_only.png")

cat("\nAUC (Width Only):", round(roc_width$auc, 4), "\n")
cat("Optimal threshold:", round(optimal_threshold, 3), "\n")

# ============================================================================
# KẾT LUẬN
# ============================================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("CONCLUSION\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\nBased on the comparison:\n")
cat("  - Height is NOT statistically significant (p = 0.166 > 0.05)\n")
cat("  - Width IS statistically significant (p < 0.001)\n")
cat("  - Width-only model has LOWER AIC (660.36 vs 660.84)\n")
cat("  - Both models have similar accuracy\n\n")

cat("✓ RECOMMENDATION: Use Width Only model (simpler, better AIC)\n")