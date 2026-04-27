library(pROC)

# 1. LOAD CLEANED DATA
data <- read.csv("clean_data_for_analysis.csv")

# 2. TRAIN/TEST SPLIT (70/30)
set.seed(123)
idx_0 <- which(data$IsAds == 0)
idx_1 <- which(data$IsAds == 1)
train_idx <- c(sample(idx_0, 0.7 * length(idx_0)), sample(idx_1, 0.7 * length(idx_1)))

train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

# 3. TRAIN LOGISTIC REGRESSION MODELS
# Model 1: Predict IsAds using the 'Width' variable
model_width <- glm(IsAds ~ Width, data = train_data, family = binomial)

# Model 2: Predict IsAds using the 'Ratio' (Aspect Ratio) variable
model_ratio <- glm(IsAds ~ Aspect_Ratio, data = train_data, family = binomial)

# 4. DETERMINE OPTIMAL THRESHOLDS
# Function to calculate the best threshold based on Youden's J statistic (Train data)
get_best_th <- function(model, train_df) {
  probs <- predict(model, train_df, type = "response")
  res_roc <- roc(train_df$IsAds, probs, quiet = TRUE)
  as.numeric(coords(res_roc, "best", ret = "threshold", transpose = TRUE))
}

th_width <- get_best_th(model_width, train_data)
th_ratio <- get_best_th(model_ratio, train_data)

# 5. PREDICT ON TEST DATA
# Generate prediction probabilities
prob_w <- predict(model_width, test_data, type = "response")
prob_r <- predict(model_ratio, test_data, type = "response")

# Convert probabilities to binary classes based on the calculated thresholds
pred_w <- ifelse(prob_w > th_width, 1, 0)
pred_r <- ifelse(prob_r > th_ratio, 1, 0)

# 6. EXPORT RESULTS
# Save the model objects as .rds files to hand over to Member 4 for final evaluation
saveRDS(model_width, "model_member2_width.rds")
saveRDS(model_ratio, "model_member2_ratio.rds")

# Print preliminary evaluation metrics to the console
cat("=== MEMBER 2: PRELIMINARY RESULTS ===\n")
cat("Optimal Threshold (Width):", round(th_width, 4), "\n")
cat("Test Accuracy (Width):    ", round(mean(pred_w == test_data$IsAds), 4), "\n\n")

cat("Optimal Threshold (Ratio):", round(th_ratio, 4), "\n")
cat("Test Accuracy (Ratio):    ", round(mean(pred_r == test_data$IsAds), 4), "\n")