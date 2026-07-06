import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import time

# 載入數據
print("正在載入MNIST數據...")
data = pd.read_csv('/ssd6/cjk0601/SC/MNIST_train.csv')  # 將檔名更改為您的CSV檔名
X = data.iloc[:, 1:].values  # 特徵（784個像素）
y = data.iloc[:, 0].values   # 標籤（數字0-9）

# 為了提高效率，可以隨機抽取部分樣本
# 對於t-SNE和UMAP，使用全部數據可能會非常耗時
n_samples = len(X)
sample_size = min(2000, n_samples)  # 最多使用2000個樣本，可以根據需要調整
if n_samples > sample_size:
    indices = np.random.choice(n_samples, sample_size, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
else:
    X_sampled = X
    y_sampled = y

print(f"使用 {len(X_sampled)} 個樣本進行降維分析")

# 標準化數據
print("正在標準化數據...")
scaler = StandardScaler()
X_std = scaler.fit_transform(X_sampled)

# 1. PCA降維
print("正在執行PCA降維...")
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
pca_time = time.time() - start_time
print(f"PCA完成，用時 {pca_time:.2f} 秒")
print(f"PCA解釋方差比例: {pca.explained_variance_ratio_}")

# 2. t-SNE降維
print("正在執行t-SNE降維...")
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_std)
tsne_time = time.time() - start_time
print(f"t-SNE完成，用時 {tsne_time:.2f} 秒")

# 3. UMAP降維
print("正在執行UMAP降維...")
start_time = time.time()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_std)
umap_time = time.time() - start_time
print(f"UMAP完成，用時 {umap_time:.2f} 秒")

# 可視化比較
print("正在生成可視化結果...")
plt.figure(figsize=(18, 6))

# 顏色映射，確保各個數字使用不同顏色
cmap = plt.cm.get_cmap('tab10', 10)

# PCA結果
plt.subplot(131)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sampled, cmap=cmap, alpha=0.8, s=10)
# plt.title(f'PCA (用時: {pca_time:.2f}秒)\n解釋方差: {sum(pca.explained_variance_ratio_):.2%}')
# plt.colorbar(scatter, ticks=range(10), label='數字類別')
plt.title(f'PCA (Time: {pca_time:.2f}s)\nExplained Variance: {sum(pca.explained_variance_ratio_):.2%}')
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.grid(True, linestyle='--', alpha=0.7)

# t-SNE結果
plt.subplot(132)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sampled, cmap=cmap, alpha=0.8, s=10)
# plt.title(f't-SNE (用時: {tsne_time:.2f}秒)')
# plt.colorbar(scatter, ticks=range(10), label='數字類別')
plt.title(f't-SNE (Time: {tsne_time:.2f}s)')
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.grid(True, linestyle='--', alpha=0.7)

# UMAP結果
plt.subplot(133)
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_sampled, cmap=cmap, alpha=0.8, s=10)
# plt.title(f'UMAP (用時: {umap_time:.2f}秒)')
# plt.colorbar(scatter, ticks=range(10), label='數字類別')
plt.title(f'UMAP (Time: {umap_time:.2f}s)')
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('mnist_dimensionality_reduction_comparison.png', dpi=300)
plt.show()

# 進一步分析：計算各個聚類的分離程度
from sklearn.metrics import silhouette_score

try:
    pca_silhouette = silhouette_score(X_pca, y_sampled)
    tsne_silhouette = silhouette_score(X_tsne, y_sampled)
    umap_silhouette = silhouette_score(X_umap, y_sampled)
    
    print("\n聚類質量評估 (Silhouette Score, 越高越好):")
    print(f"PCA: {pca_silhouette:.4f}")
    print(f"t-SNE: {tsne_silhouette:.4f}")
    print(f"UMAP: {umap_silhouette:.4f}")
except:
    print("無法計算Silhouette分數，可能樣本數過少或類別不平衡")

# 輸出性能比較
print("\n性能比較:")
print(f"{'方法':<10}{'計算時間(秒)':<15}{'備註'}")
print("-" * 60)
print(f"{'PCA':<10}{pca_time:<15.2f}解釋方差: {sum(pca.explained_variance_ratio_):.2%}")
print(f"{'t-SNE':<10}{tsne_time:<15.2f}非線性，保留局部結構")
print(f"{'UMAP':<10}{umap_time:<15.2f}非線性，平衡局部與全局結構")