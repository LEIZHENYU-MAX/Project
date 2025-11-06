# ============================================================
# Step 1: Load clean data and run PCA (no plots yet)
# ============================================================

# 1) Read the cleaned Delhi dataset
df <- read.csv("data/delhi_clean.csv")

# 2) Separate features (X) and label (y)
#    Keep only numeric pollutant columns for PCA; keep AQI_Bucket for later use
num_cols <- c("PM2.5","PM10","NO","NO2","NOx","NH3",
              "CO","SO2","O3","AQI")
X <- df[ , num_cols]
y <- df$AQI_Bucket

# 3) Standardize features (PCA requires comparable scales)
#    scale() standardizes each column (mean=0, sd=1)
X_scaled <- scale(X)

# 4) Run PCA with prcomp on standardized data
#    prcomp() returns principal components and variance
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

# 1) Basic scree plot (individual variance ratio)
plot(var_ratio, 
     type = "b",                      # both points and lines
     pch = 19,                        # solid circle
     col = "steelblue",
     xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     main = "Scree Plot of PCA")

# 2) Add cumulative variance plot
plot(cum_var_ratio,
     type = "b",
     pch = 19,
     col = "darkorange",
     xlab = "Number of Principal Components",
     ylab = "Cumulative Proportion of Variance Explained",
     main = "Cumulative Variance Explained")
abline(h = 0.8, col = "red", lty = 2)   # 80% threshold line

# ============================================================
# Step 3: Visualize PCA results (biplot of PC1 vs PC2)
# ============================================================

# 1) Create a standard biplot using base R
biplot(pca_fit,
       scale = 0,                  # do not re-scale arrows
       cex = 0.6,                  # text size
       col = c("gray40", "tomato"),# colors: points & arrows
       main = "PCA Biplot (PC1 vs PC2)")

# 2) Optional: visualize group separation by AQI_Bucket
#    Use ggplot2 for a cleaner scatter plot
library(ggplot2)

pca_df <- as.data.frame(pca_fit$x[, 1:2])   # first two PCs
pca_df$AQI_Bucket <- y                      # add labels

ggplot(pca_df, aes(x = PC1, y = PC2, color = AQI_Bucket)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "PCA Scatter Plot by AQI Category",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

# ============================================================
# Step 4: Train SVM model using PCA results
# ============================================================

# 1) Extract top 3 principal components for modeling
pca_data <- as.data.frame(pca_fit$x[, 1:3])
colnames(pca_data) <- c("PC1", "PC2", "PC3")

# 2) Add target variable (AQI_Bucket)
pca_data$AQI_Bucket <- y

# 3) Split data into training (70%) and testing (30%) sets
set.seed(123)  # for reproducibility
train_index <- sample(1:nrow(pca_data), 0.7 * nrow(pca_data))
train_data <- pca_data[train_index, ]
test_data  <- pca_data[-train_index, ]

# 4) Load SVM package
library(e1071)

# Convert label to factor (classification, not regression)
train_data$AQI_Bucket <- as.factor(train_data$AQI_Bucket)
test_data$AQI_Bucket  <- as.factor(test_data$AQI_Bucket)

# 5) Train an SVM model (radial basis kernel)
svm_model <- svm(AQI_Bucket ~ ., data = train_data, kernel = "radial")

# 6) Predict on test set
pred <- predict(svm_model, newdata = test_data)

# 7) Evaluate model performance
conf_mat <- table(Predicted = pred, Actual = test_data$AQI_Bucket)
print(conf_mat)

# 8) Compute overall accuracy
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat("Overall Accuracy:", round(accuracy, 4))

# ============================================================
# Step 5: Hyperparameter tuning for SVM
# ============================================================

# 1) Perform grid search for optimal cost and gamma
set.seed(123)
tuned <- tune.svm(
  AQI_Bucket ~ .,
  data = train_data,
  kernel = "radial",
  cost = 2^(0:4),      # test several cost values (1, 2, 4, 8, 16)
  gamma = 2^(-3:1)     # test several gamma values (0.125, 0.25, 0.5, 1, 2)
)

# 2) Print best parameter combination
print(tuned$best.parameters)

# 3) Train final SVM model using tuned parameters
svm_tuned <- svm(
  AQI_Bucket ~ .,
  data = train_data,
  kernel = "radial",
  cost = tuned$best.parameters$cost,
  gamma = tuned$best.parameters$gamma
)

# 4) Predict again on test data
pred_tuned <- predict(svm_tuned, newdata = test_data)

# 5) Evaluate tuned model
conf_mat_tuned <- table(Predicted = pred_tuned, Actual = test_data$AQI_Bucket)
print(conf_mat_tuned)

# 6) Compute accuracy
accuracy_tuned <- sum(diag(conf_mat_tuned)) / sum(conf_mat_tuned)
cat("Tuned Model Accuracy:", round(accuracy_tuned, 4))

# ============================================================
# Step 6: Visualize SVM decision regions on PC1–PC2 plane
# ============================================================

library(ggplot2)

# 1) Create a fine grid covering the range of PC1–PC2
x_range <- seq(min(pca_data$PC1), max(pca_data$PC1), length = 200)
y_range <- seq(min(pca_data$PC2), max(pca_data$PC2), length = 200)
grid <- expand.grid(PC1 = x_range, PC2 = y_range, PC3 = 0)  # PC3 fixed as 0 for 2D projection

# 2) Predict class for each grid point
grid$Pred <- predict(svm_tuned, newdata = grid)

# 3) Prepare test points for overlay
test_plot <- test_data

# 4) Plot classification regions + real data points
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
