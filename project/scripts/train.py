import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# 設定繪圖風格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 資料讀取與前處理
# ==============================================================================
# 請修改您的資料路徑
DATA_DIR = r"D:\CJK\114-1\statistics\final_project\store-sales-time-series-forecasting"

def load_and_prep_data(data_dir):
    print("正在讀取資料...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"))
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    
    # 日期轉換
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])
    holidays['date'] = pd.to_datetime(holidays['date'])
    
    # 篩選 2016 年以後 (加速訓練並聚焦近期趨勢)
    train = train[train['date'] >= "2016-01-01"].copy()
    
    return train, test, holidays, stores

# ==============================================================================
# 2. V4.1 特徵工程與輔助函數
# ==============================================================================
def get_holiday_feature(dates, store_nbr, holidays_df, stores_df):
    """取得特定商店的精準假日 (National + Matching Local)"""
    store_info = stores_df[stores_df['store_nbr'] == store_nbr].iloc[0]
    city, state = store_info['city'], store_info['state']
    
    mask_national = (holidays_df['locale'] == 'National')
    mask_regional = (holidays_df['locale'] == 'Regional') & (holidays_df['locale_name'] == state)
    mask_local = (holidays_df['locale'] == 'Local') & (holidays_df['locale_name'] == city)
    
    relevant = holidays_df[
        (mask_national | mask_regional | mask_local) & 
        (~holidays_df['transferred'])
    ]['date'].unique()
    
    return dates.isin(relevant).astype(int).values

def create_base_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_payday'] = df['date'].apply(lambda x: 1 if (x.day == 15 or x.is_month_end) else 0)
    df['onpromotion'] = df['onpromotion'].fillna(0)
    
    # Fourier Terms (V4 核心)
    day_of_year = df['date'].dt.dayofyear
    df['sin_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_year'] = np.cos(2 * np.pi * day_of_year / 365.25)
    df['is_new_year'] = (day_of_year == 1).astype(int)
    return df

# ==============================================================================
# 3. V4.1 Fused Lasso 模型核心
# ==============================================================================
def train_fused_lasso_v4_1(series_train, df_test, store_nbr, holidays, stores, lambda_2=2000):
    # 1. 準備 Lag-16
    min_date = series_train['date'].min()
    max_date = df_test['date'].max()
    full_series = pd.DataFrame({'date': pd.date_range(min_date, max_date)})
    
    train_subset = series_train[['date', 'sales']].set_index('date')
    full_series = full_series.merge(train_subset, on='date', how='left')
    full_series['sales'] = full_series['sales'].fillna(method='ffill').fillna(0)
    full_series['lag_16'] = full_series['sales'].shift(16).fillna(0)
    
    # 切分 Train 特徵
    train_indices = full_series['date'].isin(series_train['date'])
    test_indices = full_series['date'].isin(df_test['date'])
    
    lag_16_train = full_series.loc[train_indices, 'lag_16'].values
    lag_16_test = full_series.loc[test_indices, 'lag_16'].values
    y = series_train['sales'].values
    N = len(y)
    
    # 2. 準備假日與基礎特徵
    is_holiday_train = get_holiday_feature(series_train['date'], store_nbr, holidays, stores)
    is_holiday_test = get_holiday_feature(df_test['date'], store_nbr, holidays, stores)
    
    # 建立矩陣 X
    day_of_week = series_train['date'].dt.dayofweek.values
    X_weekday = np.zeros((N, 7))
    for i in range(N): X_weekday[i, day_of_week[i]] = 1
        
    feats_train = np.column_stack([
        series_train['onpromotion'].values,
        series_train['is_payday'].values,
        is_holiday_train,
        series_train['sin_year'].values,
        series_train['cos_year'].values,
        series_train['is_new_year'].values,
        lag_16_train
    ])
    X_matrix = np.hstack([X_weekday, feats_train])
    
    # 3. 求解 (CVXPY)
    beta_trend = cp.Variable(N)
    beta_linear = cp.Variable(X_matrix.shape[1])
    
    y_pred = beta_trend + X_matrix @ beta_linear
    
    # Time Decay Weighting
    weights = np.linspace(0.5, 1.5, N)
    loss = 0.5 * cp.sum_squares(cp.multiply(np.sqrt(weights), y - y_pred))
    reg_fusion = cp.norm(cp.diff(beta_trend), 1)
    
    prob = cp.Problem(cp.Minimize(loss + lambda_2 * reg_fusion))
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3)
    except:
        prob.solve(solver=cp.SCS)
        
    # 4. 預測 Test Set
    N_test = len(df_test)
    test_days = df_test['date'].dt.dayofweek.values
    X_test_weekday = np.zeros((N_test, 7))
    for i in range(N_test): X_test_weekday[i, test_days[i]] = 1
        
    feats_test = np.column_stack([
        df_test['onpromotion'].values,
        df_test['is_payday'].values,
        is_holiday_test,
        df_test['sin_year'].values,
        df_test['cos_year'].values,
        df_test['is_new_year'].values,
        lag_16_test
    ])
    X_test_matrix = np.hstack([X_test_weekday, feats_test])
    
    final_forecast = beta_trend.value[-1] + X_test_matrix @ beta_linear.value
    final_forecast = np.maximum(final_forecast, 0)
    
    # 回傳詳細資訊供繪圖 (包含 Training 期間的擬合值與 Trend)
    train_fitted = (beta_trend.value + X_matrix @ beta_linear.value)
    train_trend = beta_trend.value
    
    return final_forecast, train_fitted, train_trend

# ==============================================================================
# 4. 批次訓練與結果收集 (Visualization Pipeline)
# ==============================================================================
def run_full_pipeline_for_viz(train_df, test_df, holidays, stores, target_family="GROCERY I", max_stores=None):
    """
    執行訓練並收集所有店的結果，回傳一個包含所有歷史擬合數據的大表
    """
    train_df = create_base_features(train_df)
    test_df = create_base_features(test_df)
    
    store_ids = train_df['store_nbr'].unique()
    if max_stores:
        store_ids = store_ids[:max_stores] # 測試用，只跑前幾家店
        
    all_history_results = []
    
    print(f"開始批次訓練 (Family: {target_family})...")
    
    for i, store in enumerate(store_ids):
        if i % 10 == 0: print(f"Processing Store {store}...")
        
        mask_train = (train_df['store_nbr'] == store) & (train_df['family'] == target_family)
        train_sub = train_df[mask_train]
        mask_test = (test_df['store_nbr'] == store) & (test_df['family'] == target_family)
        test_sub = test_df[mask_test]
        
        if train_sub.empty: continue
            
        # 訓練模型
        pred_test, pred_train, trend_train = train_fused_lasso_v4_1(
            train_sub, test_sub, store, holidays, stores
        )
        
        # 收集 Training 期間的結果 (用於畫過去的趨勢)
        res_df = train_sub[['date', 'store_nbr', 'sales']].copy()
        res_df['pred_sales'] = pred_train
        res_df['trend_component'] = trend_train
        res_df['family'] = target_family
        
        all_history_results.append(res_df)
        
    # 合併並加入地理資訊 (State/City)
    full_results = pd.concat(all_history_results, ignore_index=True)
    full_results = full_results.merge(stores[['store_nbr', 'city', 'state', 'type', 'cluster']], on='store_nbr', how='left')
    
    return full_results

# ==============================================================================
# 5. 通用視覺化函數 (核心功能)
# ==============================================================================
def plot_fused_lasso_insights(full_results, level='store', target_id=None, save_path=None):
    """
    level: 'store' (單店), 'state' (地區加總), 'all' (全國加總)
    target_id: store_nbr (若 level='store'), state_name (若 level='state'), None (若 level='all')
    """
    
    # 1. 根據層級聚合資料
    if level == 'store':
        plot_df = full_results[full_results['store_nbr'] == target_id].copy()
        title_text = f"Single Store Analysis: Store {target_id}"
        
    elif level == 'state':
        # 篩選該地區並加總
        subset = full_results[full_results['state'] == target_id]
        plot_df = subset.groupby('date')[['sales', 'pred_sales', 'trend_component']].sum().reset_index()
        title_text = f"Regional Analysis: State {target_id} (Aggregated)"
        
    elif level == 'all':
        # 全國加總
        plot_df = full_results.groupby('date')[['sales', 'pred_sales', 'trend_component']].sum().reset_index()
        title_text = f"National Overview: All Stores (Aggregated)"
    
    if plot_df.empty:
        print("沒有資料可繪圖！請檢查輸入 ID。")
        return

    # 2. 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 上圖：真實銷量 vs 預測
    ax1.plot(plot_df['date'], plot_df['sales'], color='lightgray', label='Actual Sales', alpha=0.8)
    ax1.plot(plot_df['date'], plot_df['pred_sales'], color='#1f77b4', label='V4.1 Prediction', linewidth=1.5, alpha=0.9)
    ax1.set_title(f"{title_text} - Sales vs Prediction", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Sales Volume")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下圖：Fused Lasso 趨勢項
    ax2.plot(plot_df['date'], plot_df['trend_component'], color='#d62728', label='Fused Lasso Trend', linewidth=2.5)
    
    # 標註地震 (Earthquake) - 這是重要事件
    eq_date = pd.Timestamp("2016-04-16")
    if eq_date >= plot_df['date'].min() and eq_date <= plot_df['date'].max():
        ax2.axvline(eq_date, color='black', linestyle='--', alpha=0.6)
        ax2.text(eq_date, ax2.get_ylim()[1]*0.95, ' Earthquake', fontsize=10, fontweight='bold')
        
    ax2.set_title("Underlying Structural Trend (Piecewise Constant)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Trend Level")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"圖表已儲存: {save_path}")
    
    plt.show()

# ==============================================================================
# 6. 主執行區塊 (範例)
# ==============================================================================
if __name__ == "__main__":
    # 1. 讀取資料
    train_df, test_df, holidays, stores = load_and_prep_data(DATA_DIR)
    
    # 2. 執行批次訓練 (這裡以 Grocery I 為例，您可以換成其他 family)
    # 注意：跑所有店約需 10-20 分鐘，測試時可設 max_stores=5
    results_master = run_full_pipeline_for_viz(
        train_df, test_df, holidays, stores, 
        target_family="GROCERY I", 
        max_stores=None # 設為 None 跑全部店
    )
    
    # 3. 產出多種視覺化圖表
    
    # (A) 單店分析 (Store 1) - 看看元老店的狀況
    plot_fused_lasso_insights(results_master, level='store', target_id=1, save_path="viz_store_1.png")
    
    # (B) 地區分析 (Pichincha) - 這是首都 Quito 所在的省份，受地震影響較小
    plot_fused_lasso_insights(results_master, level='state', target_id='Pichincha', save_path="viz_state_Pichincha.png")
    
    # (C) 地區分析 (Manabi) - 這是地震震央所在的省份，應該會看到劇烈的 Trend 變化
    plot_fused_lasso_insights(results_master, level='state', target_id='Manabi', save_path="viz_state_Manabi.png")
    
    # (D) 全國總覽 - 看看整體厄瓜多的消費趨勢
    plot_fused_lasso_insights(results_master, level='all', save_path="viz_national.png")