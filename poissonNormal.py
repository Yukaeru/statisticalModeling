

import numpy as np

r_values = np.array([-1.10, -0.30, 0.50, 1.30])

lambda_values = np.exp(0.5 + r_values)

for r, lam in zip(r_values, lambda_values):
    print(f"r = {r:.2f}, λ = {lam:.2f}")