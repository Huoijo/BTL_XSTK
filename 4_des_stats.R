#read clean data from csv file
my_data <- read.csv("clean_data_for_analysis.csv", header = TRUE)

#retrieve the needed columns
des_data <- my_data[, c("Height", "Width", "Aspect_Ratio")]

#evaluate the summary statistics
mean_val <- apply(des_data, 2, mean)
sd_val <- apply(des_data, 2, sd)
median_val <- apply(des_data, 2, median)
min_val <- apply(des_data, 2, min)
max_val <- apply(des_data, 2, max)
q1_val <- apply(des_data, 2, quantile, probs = 0.25)
q3_val <- apply(des_data, 2, quantile, probs = 0.75)

#merge into a table
summary_table <- t(data.frame(
  Mean = mean_val,
  SD = sd_val,
  Median = median_val,
  Min = min_val,
  Max = max_val,
  Q1 = q1_val,
  Q3 = q3_val
))

#print summary table
print(summary_table)

#freq and proportion of ads
table(my_data$IsAds)
round(prop.table(table(my_data$IsAds)) * 100, 2)

#histograms
hist(my_data$Height,
     main = "Histogram of Height",
     xlab = "Height",
     col = "blue",
     border = "black")

hist(my_data$Width,
     main = "Histogram of Width",
     xlab = "Width",
     col = "orange",
     border = "black")

hist(my_data$Aspect_Ratio,
     main = "Histogram of Aspect Ratio",
     xlab = "Aspect Ratio",
     col = "darkred",
     border = "black")

#boxplots by ads
boxplot(Height ~ IsAds,
        data = my_data,
        main = "Boxplot of Height by IsAds",
        xlab = "IsAds (0 = Non-Ad, 1 = Ad)",
        ylab = "Height",
        col = c("red", "green"))

boxplot(Width ~ IsAds,
        data = my_data,
        main = "Boxplot of Width by IsAds",
        xlab = "IsAds (0 = Non-Ad, 1 = Ad)",
        ylab = "Width",
        col = c("lightblue", "gray"))

boxplot(Aspect_Ratio ~ IsAds,
        data = my_data,
        main = "Boxplot of Aspect Ratio by IsAds",
        xlab = "IsAds (0 = Non-Ad, 1 = Ad)",
        ylab = "Aspect Ratio",
        col = c("purple", "pink"))

#scatter plot
library(ggplot2)
library(httpgd)
hgd()
hgd_browse()
#scatter plot and logistic curve for Height
ggplot(my_data, aes(x = Height, y = IsAds)) +
  geom_jitter(height = 0.03, alpha = 0.35) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              se = FALSE, color = "blue") +
  labs(title = "Logistics Fit: Probability (IsAds = 1) vs Height",
       x = "Height", y = "P(IsAds = 1)") + theme_minimal()

#scatter plot and logistic curve for Width
ggplot(my_data, aes(x = Width, y = IsAds)) +
  geom_jitter(height = 0.03, alpha = 0.35) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              se = FALSE, color = "darkgreen") +
  labs(title = "Logistics Fit: Probability (IsAds = 1) vs Width",
       x = "Width", y = "P(IsAds = 1)") + theme_minimal()

#scatter plot and logistic curve for Aspect Ratio
ggplot(my_data, aes(x = Aspect_Ratio, y = IsAds)) +
  geom_jitter(height = 0.03, alpha = 0.35) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              se = FALSE, color = "red") +
  labs(title = "Logistics Fit: Probability (IsAds = 1) vs Aspect Ratio",
       x = "Aspect Ratio", y = "P(IsAds = 1)") + theme_minimal()

#correlation of predictors
library(corrplot)
library(httpgd)
hgd()
hgd_browse()

cor_mat <- cor(my_data[, c("Height", "Width", "Aspect_Ratio")])
corrplot(cor_mat,
         method = "square",
         type = "upper",
         col = COL2("RdYlBu", 200),
         addCoef.col = "black",
         tl.col = "black",
         number.cex = 0.8,
         title = "Correlation Matrix of Predictors",
         mar = c(0, 0, 1, 0))
