import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import scipy.special

# ======================
# 1. CSV読み込み・前処理
# ======================
df = pd.read_csv("data.csv")  # "N","y","x","id" を持つ

rows = []
for _, row in df.iterrows():
    for val in [1] * int(row["y"]) + [0] * int(row["N"] - row["y"]):
        rows.append({"x": row["x"], "y": val, "id": row["id"]})
df_long = pd.DataFrame(rows)

# ======================
# 2. GLMMフィット → パラメータ取得
# ======================
random = {"id": "0 + C(id)"}
glmm = BinomialBayesMixedGLM.from_formula("y ~ x", random, df_long)
result = glmm.fit_vb()

BETA0 = result.fe_mean[0]
BETA1 = result.fe_mean[1]

# ランダム効果の標準偏差（VBの事後分散から取得）
# vb_sd の最初の2つが固定効果、残りがランダム効果
# re_sd   = result.vb_sd[2:]     # 各idのランダム効果の事後標準偏差
# SIGMA   = float(np.mean(re_sd)) # 平均を代表値として使用
SIGMA = float(result.vc_sd[0])  # vc_sd はランダム効果のグループごとのSD

print(f"β̂₀  = {BETA0:.4f}")
print(f"β̂₁  = {BETA1:.4f}")
print(f"σ̂   = {SIGMA:.4f}")

# ======================
# 3. パラメータ設定
# ======================
TRUE_BETA0 = -4.0
TRUE_BETA1 =  1.0
TRUE_SIGMA =  2.0   # データ生成時の真のσに書き換える
N          = int(df["N"].iloc[0])

# ======================
# 4. 混合二項分布の周辺確率（数値積分）
# ======================
def mixed_binom_prob(k, x, beta0, beta1, sigma, N):
    """
    P(Y=k | x) = ∫ Binom(k|N, p(u)) * N(u|0,σ²) du
    p(u) = logistic(beta0 + beta1*x + u)
    """
    def integrand(u):
        p  = 1 / (1 + np.exp(-(beta0 + beta1 * x + u)))
        lk = scipy.special.comb(N, k) * p**k * (1 - p)**(N - k)
        return lk * norm.pdf(u, 0, sigma)

    val, _ = integrate.quad(integrand, -10 * sigma, 10 * sigma)
    return val

# ======================
# 5. x=4 での予測と観測を比較
# ======================
X_VAL    = 4
df_x4    = df.query("x == @X_VAL")
n_sample = df_x4.shape[0]

# 観測度数
y_vals     = np.arange(0, N + 1)
obs_counts = np.array([(df_x4["y"] == k).sum() for k in y_vals])

# GLMMの混合分布予測（個体数スケール）
pred_glmm = np.array([
    mixed_binom_prob(k, X_VAL, BETA0, BETA1, SIGMA, N) * n_sample
    for k in y_vals
])

# 真の混合分布（個体数スケール）
pred_true = np.array([
    mixed_binom_prob(k, X_VAL, TRUE_BETA0, TRUE_BETA1, TRUE_SIGMA, N) * n_sample
    for k in y_vals
])

# ======================
# 6. プロット
# ======================
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(y_vals, obs_counts,
        color="gray", linewidth=1.5,
        marker="o", markersize=8,
        label="観測データ（x=4）")

ax.plot(y_vals, pred_true,
        color="green", linewidth=1.8, linestyle="--",
        marker="o", markersize=7,
        markerfacecolor="white", markeredgewidth=2,
        label=f"真の混合分布  (σ={TRUE_SIGMA})")

ax.plot(y_vals, pred_glmm,
        color="steelblue", linewidth=1.8,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgewidth=2,
        label=f"GLMMの予測  (σ̂={SIGMA:.3f})")

ax.set_xlabel("生存種子数 $y_i$", fontsize=13)
ax.set_ylabel("個体数", fontsize=13)
ax.set_title(f"x = {X_VAL}：混合二項分布による予測  (N={N}, n={n_sample})", fontsize=14)
ax.set_xticks(y_vals)
ax.set_ylim(-0.5, obs_counts.max() + 1.5)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("glmm_mixed_binom_x4.png", dpi=150)
plt.show()