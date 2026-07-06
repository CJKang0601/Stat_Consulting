library(ggplot2)
library(dplyr)      # 用於資料整理
library(lubridate)  # 用於處理日期
library(genlasso)   # 用於 Fused Lasso
library(glmnet)     # 用於 Standard Lasso
library(tidyr)
# ==============================================================================
# 2. 讀取與篩選資料 (Data Preprocessing)
# ==============================================================================

file_path <- "D:\\CJK\\114-1\\statistics\\final_project\\store-sales-time-series-forecasting\\train.csv"

cat("正在讀取資料，請稍候... (檔案較大)\n")
raw_data <- read.csv(file_path)

# 將日期欄位轉為 Date 格式
raw_data$date <- as.Date(raw_data$date)

total_sales_trend <- raw_data %>%
  group_by(date) %>%
  summarise(total_sales = sum(sales))

p1 <- ggplot(total_sales_trend, aes(x = date, y = total_sales)) +
  geom_line(color = "steelblue", size = 0.8) +
  labs(title = "Total Daily Sales Over Time (All Stores)",
       subtitle = "Observe the seasonality and end-of-year spikes",
       y = "Total Sales", x = "Date") +
  theme_minimal(16)

print(p1)
ggsave("Total_daily_sales_Plot.png", plot = p1, width = 10, height = 6, dpi = 300)

top_families <- raw_data %>%
  group_by(family) %>%
  summarise(total_sales = sum(sales)) %>%
  arrange(desc(total_sales)) %>%
  head(10)

p2 <- ggplot(top_families, aes(x = reorder(family, total_sales), y = total_sales)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() + 
  labs(title = "Top 10 Best-Selling Product Families",
       y = "Total Sales", x = "Product Family") +
  theme_minimal(14)

print(p2)

weekly_pattern <- raw_data %>%
  mutate(weekday = wday(date, label = TRUE)) %>% # 轉換為 Mon, Tue...
  group_by(date, weekday) %>%
  summarise(daily_total = sum(sales))

p3 <- ggplot(weekly_pattern, aes(x = weekday, y = daily_total, fill = weekday)) +
  geom_boxplot() +
  labs(title = "Sales Distribution by Day of Week",
       subtitle = "Are weekends busier?",
       y = "Total Daily Sales", x = "Weekday") +
  theme_minimal(14) +
  theme(legend.position = "none")

print(p3)

# 【關鍵步驟】篩選資料 (Subsetting)
# 為了演示效果，我們選：
# Store 1 (1號店)
# Family 'GROCERY I' (雜貨類，銷售量通常比較連續)
# 時間：2016年 (取最後一年的資料，約 200-300 筆最適合展示)
target_data <- raw_data %>%
  filter(store_nbr == 1, 
         family == "GROCERY I",
         date >= as.Date("2016-01-01")) %>%
  arrange(date) # 確保依照時間排序

# 準備模型用的變數
y <- target_data$sales
dates <- target_data$date
N <- length(y)

cat("篩選後資料筆數 N =", N, "\n")
# 如果 N 超過 500，建議再縮短時間範圍，不然 Fused Lasso 會跑很久
if(N > 500) warning("資料筆數有點多，繪圖可能會比較擠。")

# ==============================================================================
# 3. 執行 Fused Lasso (Trend Filtering)
# ==============================================================================
cat("正在執行 Fused Lasso...\n")
# fusedlasso1d 預設 X 為單位矩陣 (Identity Matrix)，即 Signal Approximator
# 這裡我們不指定 lambda，讓它自己算出一整條路徑
out_fused <- fusedlasso1d(y = y)

# 選擇一個適當的 lambda (這裡選第 30 個，通常能抓到不錯的趨勢)
# 您可以試著調整這個數字 (例如 10, 30, 50) 看看平滑程度的變化
k_fused <- 30 
beta_fused <- coef(out_fused, lambda = out_fused$lambda[k_fused])$beta

# ==============================================================================
# 4. 執行 Standard Lasso (對照組)
# ==============================================================================
cat("正在執行 Standard Lasso...\n")
# 為了公平比較，我們要在同樣的基礎下跑 Lasso。
# Fused Lasso 的基礎是 Signal Approximator (y = I*beta + e)
# 所以這裡的 X 我們設為單位矩陣 (Diagonal Matrix)
X_identity <- diag(N) 

out_lasso <- glmnet(x = X_identity, y = y, alpha = 1, standardize = FALSE)

# 選擇一個 lambda (選跟 Fused Lasso 差不多數量的非零係數，或直接取路徑中間)
# 這裡取第 30 個
k_lasso <- 30
beta_lasso <- as.numeric(predict(out_lasso, newx = X_identity, s = out_lasso$lambda[k_lasso]))

# ==============================================================================
# 5. 繪圖比較 (Visualization)
# ==============================================================================
# 建立繪圖資料框
plot_df <- data.frame(
  Date = dates,
  Original_Sales = y,
  Fused_Lasso = as.numeric(beta_fused),
  Standard_Lasso = beta_lasso
)

# 使用 ggplot2 畫圖
p <- ggplot(plot_df, aes(x = Date)) +
  # 1. 原始資料 (灰色點) - 代表真實銷售量
  geom_point(aes(y = Original_Sales, color = "Original Data"), alpha = 0.4, size = 1.5) +
  
  # 2. Standard Lasso (藍色虛線) - 對照組
  geom_line(aes(y = Standard_Lasso, color = "Standard Lasso"), linewidth = 0.8, linetype = "dotted") +
  
  # 3. Fused Lasso (紅色實線) - 我們的主角
  geom_line(aes(y = Fused_Lasso, color = "Fused Lasso (Trend)"), linewidth = 1.2) +
  
  # 設定顏色
  scale_color_manual(values = c(
    "Original Data" = "gray60",
    "Standard Lasso" = "blue",
    "Fused Lasso (Trend)" = "red"
  )) +
  
  # 設定標題與標籤
  labs(title = "Comparison: Fused Lasso vs. Standard Lasso on Store Sales",
       subtitle = paste("Store 1, Grocery I (2016), N =", N),
       y = "Daily Sales",
       x = "Date",
       color = "Model") +
  
  # 主題設定
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1) # 日期轉個角度才不會疊在一起
  )

# 顯示圖片
print(p)

# 儲存圖片
ggsave("Sales_Comparison_Plot.png", plot = p, width = 10, height = 6, dpi = 300)
cat("圖片已儲存至: Sales_Comparison_Plot.png\n")
