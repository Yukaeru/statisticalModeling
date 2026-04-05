import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# パラメータ
N = 8
sigma_r = 3
n_samples = 100000

# ランダム効果 r ~ N(0,3^2)
r_samples = np.random.normal(0, sigma_r, n_samples)

# q = logistic(r)
q_samples = 1 / (1 + np.exp(-r_samples))

# y = 0〜8 の周辺確率
y_values = np.arange(0, N + 1)
p_y = np.array([np.mean(binom.pmf(y, N, q_samples)) for y in y_values])

# 点＋直線補完
plt.plot(y_values, p_y, marker='o', linestyle='-')

plt.xlabel("y")
plt.ylabel("P(y)")
plt.title("Binomial GLMM (N=8, r ~ N(0,3²))")
plt.grid(True)
plt.show()