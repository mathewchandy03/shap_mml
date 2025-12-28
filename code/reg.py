import numpy as np
from shap_mml import ShapMML
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import r2_score

# -------------------------------
# Synthetic Regression Generator
# -------------------------------

def generate_synthetic_regression(
    n=1000,
    n_modalities=10,
    attrs_per_modality=3,
    epsilon=0.0,
    seed=0
):
    rng = np.random.default_rng(seed)

    d = n_modalities * attrs_per_modality

    # Construct block-diagonal A
    A = np.zeros((d, d))
    for m in range(n_modalities):
        block = rng.uniform(-1, 1, size=(attrs_per_modality, attrs_per_modality))
        block = block @ block.T
        block /= np.max(np.abs(block))
        idx = slice(m * attrs_per_modality, (m + 1) * attrs_per_modality)
        A[idx, idx] = block

    # Construct off-diagonal B
    B = rng.uniform(-1, 1, size=(d, d))
    B = B @ B.T
    B = B / np.max(np.abs(B))

    cov = (1 - epsilon) * A + epsilon * B

    X = rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)

    beta = rng.uniform(-1, 1, size=d)
    alpha = rng.uniform(-1, 1)
    noise = rng.normal(0, 1, size=n)

    Y = X @ beta + alpha + noise

    modalities = {
        m: list(range(m * attrs_per_modality, (m + 1) * attrs_per_modality))
        for m in range(n_modalities)
    }

    return X, Y, modalities

# -------------------------------
# Learning & Prediction Functions
# -------------------------------

def linear_learning_fn(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]

def linear_predict_fn(X, model):
    return X @ model

# -------------------------------
# Experiment
# -------------------------------

X, Y, MODALITIES = generate_synthetic_regression(epsilon=0.2)

shapmml_model = ShapMML(
    x=X[:500],
    y=Y[:500],
    modalities=MODALITIES,
    learning_fn=linear_learning_fn,
    predict_fn=linear_predict_fn,
    loss_fn=lambda y, yhat: (y - yhat) ** 2,
    task_type="regression"
)

shapmml_model.train()
shapmml_model.marginal_calibrate()

print(shapmml_model.significance_table)

p = shapmml_model.p
num_modalities_list = np.arange(0, p+1)
test_mse_list = []
test_r2_list = []
counts_list = []

X_test = X[500:]
Y_true_test = Y[500:]

for q in num_modalities_list:
    # 3. Tune lambdas for fixed q
    lam1_opt, lam2_opt = shapmml_model.tune_lambdas(
        lambda1_grid=[1e-3, 1e-2, 1e-1, 1],
        lambda2_grid=[1e-3, 1e-2, 1e-1, 1],
        q=q,
        dim_reduce="svd"
    )

    # 4. Refit conditional quantile model on full calibration set
    shapmml_model.lambda1 = lam1_opt
    shapmml_model.lambda2 = lam2_opt
    shapmml_model.conditional_calibrate(q, dim_reduce="svd")

    # 5. Select modalities and predict
    
    Y_pred_optimal, selected_modalities_list = shapmml_model.predict_optimal_modalities(X_test)
    
    # Test MSE
    mse_val = np.mean((Y_true_test - Y_pred_optimal)**2)
    test_mse_list.append(mse_val)

    # Test R2
    r2_val = r2_score(Y_true_test, Y_pred_optimal)
    test_r2_list.append(r2_val)
    
    # Count selected modality sets
    selected_counts = Counter(selected_modalities_list)
    counts_list.append(selected_counts)

# Plot
plt.figure(figsize=(7,5))
plt.plot(num_modalities_list, test_mse_list, marker='o', linestyle='-', color='b', label='Test MSE')
plt.xticks(num_modalities_list)
plt.xlabel("Max Number of Modalities (q)")
plt.ylabel("Test MSE")
plt.title("Model Selection Path")



plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("../output/msp_reg_mse.pdf")
plt.show()

plt.figure(figsize=(7,5))
plt.plot(num_modalities_list, test_r2_list, marker='o', linestyle='-', color='b', label='Test R2')
plt.xticks(num_modalities_list)
plt.xlabel("Max Number of Modalities (q)")
plt.ylabel("Test R2")
plt.title("Model Selection Path")



plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("../output/msp_reg_r2.pdf")
plt.show()