library(genlasso)
library(ggplot2)
library (glmnet)
set.seed(42)

N <- 20    # samples
p <- 100   # features
sigma <- 0.75 # noise standard error

true_beta <- rep(0, p)
true_beta[10:20] <- 4    # 第一個區塊
true_beta[50:60] <- 3    # 第二個區塊
true_beta[90:95] <- 4.5  # 第三個區塊
X <- matrix(rnorm(N * p), nrow = N, ncol = p)
#X
y <- X %*% true_beta + rnorm(N, mean = 0, sd = sigma)

cat("X dim:", dim(X), "\n")
cat("y length:", length(y), "\n")

# Lasso 
lasso_fit <- glmnet(X, y, alpha = 1)
k_lasso <- 30
estimated_beta_lasso <- as.numeric(coef(lasso_fit)[-1, k_lasso]) # [-1] 是為了去掉 Intercept

# 3. 整理繪圖資料
plot_data_lasso <- data.frame(
  Index = 1:p,
  True_Beta = true_beta,
  Estimated_Beta = estimated_beta_lasso
)

# 4. 畫出 Lasso 的結果圖
lasso_p <- ggplot(plot_data_lasso, aes(x = Index)) +
  # 真實係數 (黑色實線)
  geom_line(aes(y = True_Beta, color = "True Beta"), linewidth = 1.2) +
  
  # Lasso 估計係數 (藍色虛線 - 換個顏色區分)
  geom_line(aes(y = Estimated_Beta, color = "Lasso Estimate"), 
            linewidth = 0.8, linetype = "dashed") +
  
  # Lasso 估計係數 (藍色點)
  geom_point(aes(y = Estimated_Beta, color = "Lasso Estimate"), 
             size = 1.5) +
  
  # 設定顏色與標題
  scale_color_manual(values = c("True Beta" = "black", "Lasso Estimate" = "blue")) +
  labs(title = "Standard Lasso Simulation ",
       subtitle = paste("N =", N, ", p =", p, ", sigma =", sigma),
       y = "Coefficient Value",
       x = "Predictor Index",
       color = "Legend") +
  theme_minimal(14) +
  theme(legend.position = "top")

ggsave("lasso_replication.png", plot = lasso_p, width = 8, height = 5, dpi = 300)

# Fused lasso simulated experiment
# 步驟 1: 提取係數並強制轉為數值向量 (關鍵修正：加上 as.numeric)
# 這樣可以避免矩陣格式導致 data.frame 欄位名稱跑掉的問題
out <- fusedlasso1d(y = y, X = X) # 特徵遠大於樣本，為了要能讓計算順利自動加入極小的 ridge penalty
k <- 30
estimated_beta <- as.numeric(coef(out, lambda = out$lambda[k])$beta)


plot_data <- data.frame(
  Index = 1:p,
  True_Beta = true_beta,
  Estimated_Beta = estimated_beta
)


fusedlasso_p <- ggplot(plot_data, aes(x = Index)) +
  # 真實係數 (黑色實線)
  geom_line(aes(y = True_Beta, color = "True Beta"), linewidth = 1.2) +
  
  # 估計係數 (紅色虛線) 連接點跟點
  geom_line(aes(y = Estimated_Beta, color = "Fused Lasso Estimate"),
            linewidth = 0.8, linetype = "dashed") +
  
  # 估計係數 (紅色點) - 注意：這裡改回使用 size
  geom_point(aes(y = Estimated_Beta, color = "Fused Lasso Estimate"), 
             size = 1.5) +
  
  # 設定顏色與標題
  scale_color_manual(values = c("True Beta" = "black", "Fused Lasso Estimate" = "red")) +
  labs(title = "Replication of Fused Lasso Simulation",
       subtitle = paste("N =", N, ", p =", p, ", sigma =", sigma),
       y = "Coefficient Value",
       x = "Predictor Index",
       color = "Legend") +
  theme_minimal(14) +
  theme(legend.position = "top")

ggsave("fusedlasso_replication.png", plot = fusedlasso_p, width = 8, height = 5, dpi = 300)





