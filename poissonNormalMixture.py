import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# パラメータ
beta = 0.5          # 固定効果
sigma_r = 1.0       # ランダム効果の標準偏差
n_samples = 100000  # r_i のモンテカルロサンプル数
y_max = 15          # 描画するyの最大値

# r_i を標準正規分布からサンプリング
r_samples = np.random.normal(0, sigma_r, size=n_samples)

# λ_i = exp(beta + r_i)
lambda_samples = np.exp(beta + r_samples)

# y = 0,1,...,y_max の周辺確率を近似
y_values = np.arange(0, y_max + 1)
p_y = np.array([np.mean(poisson.pmf(y, lambda_samples)) for y in y_values])

# 結果を表示
for y, p in zip(y_values, p_y):
    print(f"y = {y}, P(y) ≈ {p:.4f}")

# グラフ表示
plt.bar(y_values, p_y, alpha=0.6)
plt.xlabel("y")
plt.ylabel("P(y)")
plt.title("Poisson GLMM (beta=0.5, r ~ N(0,1))")
plt.show()