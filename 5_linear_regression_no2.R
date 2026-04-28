# Install necessary packages if they are not already installed
if (!require("pROC")) install.packages("pROC")
if (!require("ggplot2")) install.packages("ggplot2")

library(pROC)
library(ggplot2)

# 1. LOAD PREPROCESSED DATA (Provided by Member 3)
data <- read.csv("clean_data_for_analysis.csv")

# 2. TRAIN/TEST SPLIT (Need synchronized with Member 1's seed for consistency)
set.seed(123) #NEED FIX
idx_0 <- which(data$IsAds == 0)
idx_1 <- which(data$IsAds == 1)
train_idx <- c(sample(idx_0, 0.7 * length(idx_0)), 
               sample(idx_1, 0.7 * length(idx_1)))

train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

# 3. BUILD MODELS
# Rebuild Model 1 (Width) from Member 1 to obtain metrics for comparison
model1 <- glm(IsAds ~ Width, data = train_data, family = binomial) #NEED FIX

# Train Model 2 using Aspect Ratio
model2 <- glm(IsAds ~ Aspect_Ratio, data = train_data, family = binomial)

# 4. FIND OPTIMAL THRESHOLDS AND MAKE PREDICTIONS
# Function to find the optimal threshold using Youden's J statistic on Train data
get_best_th <- function(model, train_df) {
  probs <- predict(model, train_df, type = "response")
  roc_obj <- roc(train_df$IsAds, probs, quiet = TRUE)
  as.numeric(coords(roc_obj, "best", ret = "threshold", transpose = TRUE)[1])
}

th1 <- get_best_th(model1, train_data)
th2 <- get_best_th(model2, train_data)

# Calculate probabilities and classify on Test data
prob1 <- predict(model1, test_data, type = "response")
prob2 <- predict(model2, test_data, type = "response")

pred1 <- ifelse(prob1 > th1, 1, 0)
pred2 <- ifelse(prob2 > th2, 1, 0)

# 5. CALCULATE EVALUATION METRICS: ACCURACY, AUC, AIC
# Calculate Accuracy
acc1 <- mean(pred1 == test_data$IsAds)
acc2 <- mean(pred2 == test_data$IsAds)

# Create ROC objects and calculate AUC
roc1 <- roc(test_data$IsAds, prob1, quiet = TRUE)
roc2 <- roc(test_data$IsAds, prob2, quiet = TRUE)

auc1 <- as.numeric(auc(roc1))
auc2 <- as.numeric(auc(roc2))

# Extract AIC directly from the glm models
aic1 <- AIC(model1)
aic2 <- AIC(model2)

# COMPARE AND CONCLUDE THE BETTER MODEL
cat("\n=== PERFORMANCE COMPARISON (MEMBER 2) ===\n")
cat(sprintf("%-18s | %-10s | %-10s | %-10s\n", "Model", "Accuracy", "AUC", "AIC"))
cat("----------------------------------------------------------\n")
cat(sprintf("%-18s | %-10.4f | %-10.4f | %-10.2f\n", "Model 1 (Width)", acc1, auc1, aic1))
cat(sprintf("%-18s | %-10.4f | %-10.4f | %-10.2f\n", "Model 2 (Ratio)", acc2, auc2, aic2))
cat("----------------------------------------------------------\n")

cat("\n=== FINAL MODEL SELECTION ===\n")
# Criteria: Higher AUC, Higher Accuracy, and LOWER AIC indicates a better model
if(auc2 > auc1 && aic2 < aic1) {
  cat("=> MODEL 2 (Aspect Ratio) explains and predicts better than Model 1.\n")
} else if (auc1 > auc2 && aic1 < aic2) {
  cat("=> MODEL 1 (Width) explains and predicts better than Model 2.\n")
} else {
  cat("=> Both models show a trade-off. Further practical consideration is needed.\n")
}

# PLOT ROC CURVE SPECIFICALLY FOR MODEL 2
roc_plot_m2 <- ggroc(roc2, color = "#2E86C1", size = 1, legacy.axes = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "darkgrey") +
  labs(title = "ROC Curve - Model 2 (Aspect Ratio)",
       x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)") +
  theme_minimal()

# Display the plot in the R environment
print(roc_plot_m2)

# Plot Combined ROC Curves for Comparison
roc_list <- list(
  "Model 1 (Width)" = roc1,
  "Model 2 (Ratio)" = roc2
)

# Generate the combined ROC plot using legacy.axes for "1 - Specificity"
combined_roc_plot <- ggroc(roc_list, size = 1, legacy.axes = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "darkgrey") +
  scale_color_manual(values = c("Model 1 (Width)" = "#E74C3C",  # Red for Width
                                "Model 2 (Ratio)" = "#2E86C1")) + # Blue for Ratio
  labs(title = "ROC Curves Comparison: Model 1 vs Model 2",
       subtitle = paste("AUC Model 1:", round(auc1, 4), "| AUC Model 2:", round(auc2, 4)),
       x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)",
       color = "Classification Models") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14))

# Display the combined plot in RStudio
print(combined_roc_plot)