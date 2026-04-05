import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# パラメータ
beta = 0.5          # 固定効果
sigma_r = 1.0       # ランダム効果の標準偏差
n_samples = 100000  # r_i のモンテカルロサンプル数
y_max = 15          # 描画するyの最大値

# ランダム効果のサンプル
r_samples = np.random.normal(0, sigma_r, n_samples)

# 条件付き Poisson の λ
lambda_samples = np.exp(beta + r_samples)

# y = 0,1,...,y_max の周辺確率
y_values = np.arange(0, y_max + 1)
p_y = np.array([np.mean(poisson.pmf(y, lambda_samples)) for y in y_values])

# プロット（点と直線補完）
plt.plot(y_values, p_y, marker='o', linestyle='-')
plt.xlabel("y")
plt.ylabel("P(y)")
plt.title("Poisson GLMM (beta=0.5, r ~ N(0,1))")
plt.grid(True)
plt.show()