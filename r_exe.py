
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOSのヒラギノフォントを使う
plt.rcParams['font.family'] = 'Hiragino Sans'

# ======================
# 1. RスクリプトでGLMM推定
# ======================
# r_script = """
# library(lme4)
# df <- read.csv("data.csv")
# model <- glmer(cbind(y, N-y) ~ x + (1|id), data=df, family=binomial)
# params <- c(fixef(model), sigma=as.numeric(VarCorr(model)$id)^0.5)
# write.csv(data.frame(t(params)), "glmm_params.csv", row.names=FALSE)
# """
r_script = """
library(glmmML)
df <- read.csv("data.csv")
model <- glmmML(cbind(y, N-y) ~ x, cluster=df$id, data=df, family=binomial, method="ghq")
beta1 <- model$coefficients[1]
beta2 <- model$coefficients[2]
s     <- model$sigma
write.csv(data.frame(beta1=beta1, beta2=beta2, sigma=s), "glmm_params.csv", row.names=FALSE)
"""

with open("glmm_fit.R", "w") as f:
    f.write(r_script)

subprocess.run(["Rscript", "glmm_fit.R"], check=True)

# ======================
# 2. 推定パラメータ読み込み
# ======================
# params = pd.read_csv("glmm_params.csv").iloc[0]
# beta1  = params["X.Intercept."]
# beta2  = params["x"]
# s      = params["sigma"]
params = pd.read_csv("glmm_params.csv").iloc[0]
beta1  = params["beta1"]
beta2  = params["beta2"]
s      = params["sigma"]

print(f"β₁ = {beta1:.4f}")
print(f"β₂ = {beta2:.4f}")
print(f"s  = {s:.4f}")

# ======================
# 3. データ読み込み
# ======================
df = pd.read_csv("data.csv")
df_count = pd.DataFrame(
    df.groupby(["x", "y"]).size(), columns=["counts"]
).reset_index()

N          = 8
X_VAL      = 4
Nsample_x4 = df.query("x == 4").shape[0]
xi         = X_VAL

# ======================
# 4. 積分関数
# ======================
def Binom(ri):
    qi = 1 / (1 + np.exp(-beta1 - beta2 * xi - ri))
    return scipy.special.binom(N, ni) * qi**ni * (1 - qi)**(N - ni)

def Gauss(ri):
    return np.exp(-ri**2 / (2 * s**2)) / np.sqrt(2 * np.pi) / s

def Prob(ri):
    return Binom(ri) * Gauss(ri)

# ======================
# 5. 混合二項分布を数値積分で計算
# ======================
list_estimated = []
for ni in range(0, N + 1):
    val = integrate.quad(Prob, -100, 100)[0] * Nsample_x4
    list_estimated.append(val)

# ======================
# 6. プロット
# ======================
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    df_count.query("x == 4")["y"],
    df_count.query("x == 4")["counts"],
    marker="o", linewidth=1.5, color="gray",
    label="データ"
)

ax.plot(
    list(range(0, N + 1)),
    list_estimated,
    marker="o", linewidth=1.8, color="steelblue",
    markerfacecolor="white", markeredgewidth=2,
    label=f"GLMMの予測  (β₁={beta1:.3f}, β₂={beta2:.3f}, s={s:.3f})"
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

print(f"β₁ = {beta1:.4f}")
print(f"β₂ = {beta2:.4f}")
print(f"s  = {s:.4f}")          # 標準偏差
print(f"s² = {s**2:.4f}")       # 分散