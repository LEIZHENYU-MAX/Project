# ============================================================
# Step 1: Load clean data and run PCA (no plots yet)
# ============================================================

# 1) Read the cleaned Delhi dataset
df <- read.csv("data/delhi_clean.csv")

# 2) Separate features (X) and label (y)
#    Keep only numeric pollutant columns for PCA; keep AQI_Bucket for later use
num_cols <- c("PM2.5","PM10","NO","NO2","NOx","NH3",
              "CO","SO2","O3")   # <-- FIXED: removed "AQI"

X <- df[ , num_cols]
y <- df$AQI_Bucket

# 3) Standardize features (PCA requires comparable scales)
X_scaled <- scale(X)

# 4) Run PCA with prcomp on standardized data
pca_fit <- prcomp(X_scaled, center = FALSE, scale. = FALSE)

# 5) Inspect eigenvalues and explained variance
eig_vals <- pca_fit$sdev^2
var_ratio <- eig_vals / sum(eig_vals)
cum_var_ratio <- cumsum(var_ratio)

# 6) Create a compact summary table for report
summary_table <- data.frame(
  PC = paste0("PC", seq_along(var_ratio)),
  Eigenvalue = round(eig_vals, 4),
  Var_Ratio = round(var_ratio, 4),
  Cum_Var_Ratio = round(cum_var_ratio, 4)
)

# 7) Print the summary
print(summary_table)

# ============================================================
# Step 2: Visualize variance explained by each principal component
# ============================================================

plot(var_ratio, type = "b", pch = 19, col = "steelblue",
     xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     main = "Scree Plot of PCA")

plot(cum_var_ratio, type = "b", pch = 19, col = "darkorange",
     xlab = "Number of Principal Components",
     ylab = "Cumulative Proportion of Variance Explained",
     main = "Cumulative Variance Explained")
abline(h = 0.8, col = "red", lty = 2)

# ============================================================
# Step 3: Visualize PCA results (biplot of PC1 vs PC2)
# ============================================================

biplot(pca_fit,
       scale = 0,
       cex = 0.6,
       col = c("gray40", "tomato"),
       main = "PCA Biplot (PC1 vs PC2)")

library(ggplot2)
pca_df <- as.data.frame(pca_fit$x[, 1:2])
pca_df$AQI_Bucket <- y

ggplot(pca_df, aes(x = PC1, y = PC2, color = AQI_Bucket)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "PCA Scatter Plot by AQI Category",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

# ============================================================
# Step 4: Train SVM model using PCA results
# ============================================================

pca_data <- as.data.frame(pca_fit$x[, 1:3])
colnames(pca_data) <- c("PC1", "PC2", "PC3")
pca_data$AQI_Bucket <- y

set.seed(123)
train_index <- sample(1:nrow(pca_data), 0.7 * nrow(pca_data))
train_data <- pca_data[train_index, ]
test_data  <- pca_data[-train_index, ]

library(e1071)

train_data$AQI_Bucket <- as.factor(train_data$AQI_Bucket)
test_data$AQI_Bucket  <- as.factor(test_data$AQI_Bucket)

svm_model <- svm(AQI_Bucket ~ ., data = train_data, kernel = "radial")

pred <- predict(svm_model, newdata = test_data)

conf_mat <- table(Predicted = pred, Actual = test_data$AQI_Bucket)
print(conf_mat)

accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat("Overall Accuracy:", round(accuracy, 4))

# ============================================================
# Step 5: Hyperparameter tuning for SVM
# ============================================================

set.seed(123)
tuned <- tune.svm(
  AQI_Bucket ~ .,
  data = train_data,
  kernel = "radial",
  cost = 2^(0:4),
  gamma = 2^(-3:1)
)

print(tuned$best.parameters)

svm_tuned <- svm(
  AQI_Bucket ~ .,
  data = train_data,
  kernel = "radial",
  cost = tuned$best.parameters$cost,
  gamma = tuned$best.parameters$gamma
)

pred_tuned <- predict(svm_tuned, newdata = test_data)

conf_mat_tuned <- table(Predicted = pred_tuned, Actual = test_data$AQI_Bucket)
print(conf_mat_tuned)

accuracy_tuned <- sum(diag(conf_mat_tuned)) / sum(conf_mat_tuned)
cat("Tuned Model Accuracy:", round(accuracy_tuned, 4))

# ============================================================
# Step 6: Visualize SVM decision regions on PC1–PC2 plane
# ============================================================

x_range <- seq(min(pca_data$PC1), max(pca_data$PC1), length = 200)
y_range <- seq(min(pca_data$PC2), max(pca_data$PC2), length = 200)
grid <- expand.grid(PC1 = x_range, PC2 = y_range, PC3 = 0)

grid$Pred <- predict(svm_tuned, newdata = grid)

test_plot <- test_data

ggplot() +
  geom_tile(data = grid, aes(x = PC1, y = PC2, fill = Pred), alpha = 0.25) +
  geom_point(data = test_plot, aes(x = PC1, y = PC2, color = AQI_Bucket), size = 1.8) +
  scale_fill_brewer(palette = "Pastel1") +
  labs(title = "SVM Decision Regions on PCA (PC1–PC2)",
       x = "Principal Component 1",
       y = "Principal Component 2",
       fill = "Predicted Class",
       color = "Actual Class") +
  theme_minimal()

