import numpy as np

# r の値
r_values = np.array([-2.20, -0.60, 1.00, 2.60])

# λ = 1 / (1 + exp(-r)))
lambda_values = 1 / (1 + np.exp(- r_values))

# 表示
for r, lam in zip(r_values, lambda_values):
    print(f"r = {r:.2f}, λ = {lam:.4f}")

# r = -2.20, λ = 0.0998
# r = -0.60, λ = 0.3543
# r = 1.00, λ = 0.7311
# r = 2.60, λ = 0.9309