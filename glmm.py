import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# ======================
# 1. データ読み込み・前処理
# ======================
df = pd.read_csv("data.csv")
df["prop"] = df["y"] / df["N"]

rows = []
for _, row in df.iterrows():
    for val in [1] * int(row["y"]) + [0] * int(row["N"] - row["y"]):
        rows.append({"x": row["x"], "y": val, "id": row["id"]})
df_long = pd.DataFrame(rows)

# ======================
# 2. GLMMフィット
# ======================
random = {"id": "0 + C(id)"}
glmm = BinomialBayesMixedGLM.from_formula("y ~ x", random, df_long)
result = glmm.fit_vb()

beta0_hat = result.fe_mean[0]
beta1_hat = result.fe_mean[1]

# ======================
# 3. 真のパラメータ（データ生成時の設定値）
# ======================
TRUE_BETA0 = -4.0   # ← データ生成時の真の切片に書き換える
TRUE_BETA1 =  1.0   # ← データ生成時の真の傾きに書き換える

# ======================
# 4. 予測曲線の計算
# ======================
x_range = np.linspace(df["x"].min(), df["x"].max(), 300)

# GLMM推定（固定効果）
pred_glmm = 1 / (1 + np.exp(-(beta0_hat + beta1_hat * x_range)))

# 真の分布
pred_true = 1 / (1 + np.exp(-(TRUE_BETA0 + TRUE_BETA1 * x_range)))

# ======================
# 5. プロット
# ======================
fig, ax = plt.subplots(figsize=(8, 5))

# 観測割合（バブルサイズ = N）
ax.scatter(
    df["x"], df["prop"],
    s=df["N"] * 10,
    alpha=0.6,
    color="steelblue",
    edgecolors="white",
    linewidths=0.5,
    zorder=3,
    label="Observed proportion (y/N)  ※bubble size ∝ N"
)

# 真の分布（破線・緑）
ax.plot(
    x_range, pred_true,
    color="green",
    linewidth=2,
    linestyle="dashed",
    label=f"True distribution  (β₀={TRUE_BETA0}, β₁={TRUE_BETA1})",
    zorder=4
)

# GLMM推定曲線（実線・赤）
ax.plot(
    x_range, pred_glmm,
    color="crimson",
    linewidth=2.5,
    label=f"GLMM predicted (fixed effect)  β̂₀={beta0_hat:.3f}, β̂₁={beta1_hat:.3f}",
    zorder=5
)

ax.set_xlabel("x", fontsize=13)
ax.set_ylabel("Proportion (y / N)", fontsize=13)
ax.set_title("GLMM vs True Distribution", fontsize=14)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("glmm_vs_true.png", dpi=150)
plt.show()

print(f"\n推定値  β̂₀ = {beta0_hat:.4f}  (真値 {TRUE_BETA0})")
print(f"推定値  β̂₁ = {beta1_hat:.4f}  (真値 {TRUE_BETA1})")
print(f"切片の誤差: {beta0_hat - TRUE_BETA0:+.4f}")
print(f"傾きの誤差: {beta1_hat - TRUE_BETA1:+.4f}")