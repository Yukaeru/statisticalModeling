import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.special

# ======================
# 1. データ読み込み
# ======================
df = pd.read_csv("data.csv")
df_count = pd.DataFrame(
    df.groupby(["x", "y"]).size(), columns=["counts"]
).reset_index()

N         = 8
X_VAL     = 4
Nsample_x4 = df.query("x == 4").shape[0]

# ======================
# 2. Rで推定されたGLMMパラメータ
# ======================
beta1 = -4.190   # 切片
beta2 =  1.005   # 葉数の係数
s     =  2.408   # 個体差の標準偏差

# ======================
# 3. 積分に使う関数（グローバル変数 beta1, beta2, s, ni, xi を参照）
# ======================
xi = X_VAL

def Binom(ri):
    qi = 1 / (1 + np.exp(-beta1 - beta2 * xi - ri))
    return scipy.special.binom(N, ni) * qi**ni * (1 - qi)**(N - ni)

def Gauss(ri):
    return np.exp(-ri**2 / (2 * s**2)) / np.sqrt(2 * np.pi) / s

def Prob(ri):
    return Binom(ri) * Gauss(ri)

# ======================
# 4. x=4 での混合二項分布を数値積分で計算
# ======================
list_estimated = []
for ni in range(0, N + 1):
    val = integrate.quad(Prob, -100, 100)[0] * Nsample_x4
    list_estimated.append(val)

# ======================
# 5. プロット
# ======================
fig, ax = plt.subplots(figsize=(9, 5))

# 観測データ
ax.plot(
    df_count.query("x == 4")["y"],
    df_count.query("x == 4")["counts"],
    marker="o", linewidth=1.5, color="gray",
    label="データ"
)

# GLMMの予測
ax.plot(
    list(range(0, N + 1)),
    list_estimated,
    marker="o", linewidth=1.8, color="steelblue",
    markerfacecolor="white", markeredgewidth=2,
    label=f"GLMMの予測  (β₁={beta1}, β₂={beta2}, s={s})"
)

ax.set_xlabel("生存種子数", fontsize=13)
ax.set_ylabel("観測された個体数", fontsize=13)
ax.set_title(f"x = {X_VAL} における生存種子数の分布", fontsize=14)
ax.set_ylim(-0.5, 6.5)
ax.set_xticks(range(0, N + 1))
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("glmm_mixed_binom_x4.png", dpi=150)
plt.show()