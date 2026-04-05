import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# ======================
# 1. CSV読み込み
# ======================
df = pd.read_csv("/Users/yuka/IdeaProjects/statisticalModeling/data.csv")

# ======================
# 2. GLM（固定効果のみ）
# ======================
df["prop"] = df["y"] / df["N"]

glm = sm.GLM(
    df["prop"],
    sm.add_constant(df["x"]),
    family=sm.families.Binomial(),
    var_weights=df["N"]
)

glm_result = glm.fit()

print("=== GLM ===")
print(glm_result.summary())

# ======================
# 3. GLMM用に0/1展開
# ======================
rows = []

for _, row in df.iterrows():
    success = [1] * int(row["y"])
    fail = [0] * int(row["N"] - row["y"])

    for val in success + fail:
        rows.append({
            "y": val,
            "x": row["x"],
            "id": row["id"]
        })

df_long = pd.DataFrame(rows)

# ======================
# 4. GLMM（random intercept）
# ======================
random = {"id": "0 + C(id)"}

glmm = BinomialBayesMixedGLM.from_formula(
    "y ~ x",
    random,
    df_long
)

glmm_result = glmm.fit_vb()

print("\n=== GLMM ===")
print(glmm_result.summary())