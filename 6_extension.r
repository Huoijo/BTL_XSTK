# 4. EXTENSION ANALYSIS: RANDOM FOREST (Full + Feature Selection)

library(randomForest)
library(pROC)
library(ggplot2)

# Prepare data (target as factor)
train_rf <- train_data
test_rf  <- test_data
train_rf$IsAds <- factor(train_rf$IsAds, levels = c(0, 1)) # 0 : Non , 1 : Ads
test_rf$IsAds  <- factor(test_rf$IsAds,  levels = c(0, 1))

# MODEL 1: Random Forest with 3 main features
set.seed(42) # fix random để chạy ko ra khác kết quả
rf_model <- randomForest(
  IsAds ~ Height + Width + Aspect_Ratio, # height cao - width rộng - aspect : tỉ lệ width / height : ads : rộng -> ratio lớn còn non ads ratio xấp xỉ 1
  data = train_rf,
  ntree = 550, # model này có 550 cây , random 550 features trong train_rf ra để học hỏi và mỗi test_rf sẽ qua 500 trees này ( output : 300 : ads & 200 nonads --> ads)
  importance = TRUE
)

cat("\n===== RANDOM FOREST MODEL (Height, Width, Aspect_Ratio) =====\n")
print(rf_model)
varImpPlot(rf_model, main = "Variable Importance - Random Forest (3 Features)")

# Predictions & Evaluation
rf_pred_class <- predict(rf_model, newdata = test_rf, type = "class") # dãy  0 1 ( = số ảnh )
rf_pred_prob  <- predict(rf_model, newdata = test_rf, type = "prob")[, 2] # bảng số ảnh x 2 : xác suất , cùng row + lại = 1
rf_acc <- mean(rf_pred_class == test_rf$IsAds) # tỉ lệ trúng là ads
rf_cm  <- table(Actual = test_rf$IsAds, Predicted = rf_pred_class) # confusion matrix
roc_rf <- roc(as.numeric(as.character(test_rf$IsAds)), rf_pred_prob) # đồ thị ROC
auc_rf <- auc(roc_rf) # 

cat("\n===== RANDOM FOREST EVALUATION =====\n")
cat("Accuracy:", round(rf_acc, 4), "\n")
cat("AUC:", round(auc_rf, 4), "\n")
cat("Confusion Matrix:\n")
print(rf_cm)

# ROC curve (3 features)
ggroc(roc_rf, color = "#E69F00", size = 1.3) +
  ggtitle("ROC Curve - Random Forest (Height, Width, Aspect_Ratio)") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste("AUC =", round(auc_rf, 4)),
           color = "#E69F00", size = 4.5) +
  geom_abline(intercept = 1, slope = 1,
              linetype = "dashed", color = "gray60") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))


# EXTENSION 2: Random Forest with Full 4–1558 Features
cat("\n===== RANDOM FOREST (FULL 4–1558 FEATURES) =====\n")

# Load full dataset
data_full <- read.csv("/kaggle/input/datasets/rykanny/xstk2-1/add.csv")[, -1]
colnames(data_full)[ncol(data_full)] <- "IsAds"
data_full <- data_full %>%
  mutate(across(everything(), ~trimws(.)))

data_full[data_full %in% c("?", "", "NA", "null", "NULL")] <- NA
# Convert target and numeric columns
data_full$IsAds <- as.factor(ifelse(data_full$IsAds == "ad.", 1, 0))
library(readr)

for (i in 1:(ncol(data_full) - 1)) {
  data_full[[i]] <- suppressWarnings(parse_number(as.character(data_full[[i]])))
  data_full[[i]][is.na(data_full[[i]])] <- 0
}
# Train/test split
set.seed(42)
idx <- sample(1:nrow(data_full), 0.7 * nrow(data_full))
train_full <- data_full[idx, ]
test_full  <- data_full[-idx, ]

# Random Forest on all features
set.seed(42)
rf_full <- randomForest(IsAds ~ ., data = train_full,
                        ntree = 350, importance = TRUE)
print(rf_full)

# Evaluate full model
rf_full_pred <- predict(rf_full, newdata = test_full, type = "prob")[, 2]
roc_full <- roc(test_full$IsAds, rf_full_pred)
auc_full <- auc(roc_full)
cat("AUC (Full Feature Model):", round(auc_full, 4), "\n")

# FEATURE SELECTION: Top 30 Important Features
imp <- importance(rf_full, type = 1)
top_vars <- names(sort(imp[, 1], decreasing = TRUE))[1:30]
cat("Top 30 important features:\n")
print(top_vars)

train_sel <- train_full[, c(top_vars, "IsAds")]
test_sel  <- test_full[, c(top_vars, "IsAds")]

set.seed(42)
rf_sel <- randomForest(IsAds ~ ., data = train_sel,
                       ntree = 350, importance = TRUE)

rf_sel_pred <- predict(rf_sel, newdata = test_sel, type = "prob")[, 2]
roc_sel <- roc(test_sel$IsAds, rf_sel_pred)
auc_sel <- auc(roc_sel)

cat("\n===== FEATURE SELECTION MODEL =====\n")
cat("AUC (Selected Features):", round(auc_sel, 4), "\n")

# VISUAL COMPARISON: ROC Full vs Selected
par(mfrow = c(1, 2))  # 1 hàng 2 cột
plot(roc_full, col = "blue", lwd = 2,
     main = "ROC - Random Forest with All Features")
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.6, 0.2, paste("AUC =", round(auc_full, 4)), col = "blue")

plot(roc_sel, col = "blue", lwd = 2,
     main = "ROC - Random Forest with Feature Selection")
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.6, 0.2, paste("AUC =", round(auc_sel, 4)), col = "blue")
par(mfrow = c(1, 1))

# SAVE ROC COMPARISON TO FILE 
png("roc_comparison_rf.png", width = 1000, height = 450)
par(mfrow = c(1, 2))
plot(roc_full, col = "blue", lwd = 2,
     main = "ROC - Random Forest with All Features")
abline(a = 0, b = 1, lty = 2, col = "gray")
plot(roc_sel, col = "blue", lwd = 2,
     main = "ROC - Random Forest with Feature Selection")
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# EXTENSION ANALYSIS: DECISION TREE MODEL
library(rpart)
library(rpart.plot)
library(pROC)
library(ggplot2)

# Prepare training and testing datasets
# (same structure as used in Logistic Regression)
train_tree <- train_data
test_tree  <- test_data
train_tree$IsAds <- factor(train_tree$IsAds, levels = c(0, 1))
test_tree$IsAds  <- factor(test_tree$IsAds,  levels = c(0, 1))

# BUILD DECISION TREE MODEL
set.seed(42)
tree_model <- rpart(IsAds ~ Height + Width + Aspect_Ratio,
                    data = train_tree,
                    method = "class",
                    control = rpart.control(cp = 0.001, minsplit = 10))

cat("\n===== DECISION TREE MODEL (Height, Width, Aspect_Ratio) =====\n")
printcp(tree_model)  # Display complexity parameter (CP) table

# Visualize the decision tree
rpart.plot(tree_model,
           main = "Decision Tree - Height, Width, Aspect_Ratio",
           box.palette = "GnBu",
           shadow.col = "gray",
           nn = TRUE)  # Show node numbers
# MODEL EVALUATION
# Predict probabilities and class labels
tree_pred_prob  <- predict(tree_model, newdata = test_tree, type = "prob")[, 2]
tree_pred_class <- ifelse(tree_pred_prob > 0.5, 1, 0)

# Calculate accuracy and confusion matrix
tree_acc <- mean(tree_pred_class == test_tree$IsAds)
tree_cm  <- table(Actual = test_tree$IsAds, Predicted = tree_pred_class)

# Generate ROC curve and AUC value
roc_tree <- roc(as.numeric(as.character(test_tree$IsAds)), tree_pred_prob)
auc_tree <- auc(roc_tree)

cat("\n===== DECISION TREE EVALUATION =====\n")
cat("Accuracy:", round(tree_acc, 4), "\n")
cat("AUC:", round(auc_tree, 4), "\n")
cat("Confusion Matrix:\n")
print(tree_cm)
# ROC CURVE VISUALIZATION
ggroc(roc_tree, color = "#CC79A7", size = 1.3) +
  ggtitle("ROC Curve - Decision Tree (Height, Width, Aspect_Ratio)") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste("AUC =", round(auc_tree, 4)),
           color = "#CC79A7", size = 4.5) +
  geom_abline(intercept = 1, slope = 1,
              linetype = "dashed", color = "gray60") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))