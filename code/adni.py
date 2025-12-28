from shap_mml import *
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import r2_score

def mse_loss_fn(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Squared Error (MSE) loss function, used for calculating utility."""
    # MSE is a common utility measure in this context (Prop. 3.4 uses squared loss)
    # The output needs to be a vector of losses (one per observation)
    return (y_true - y_pred)**2

def xgboost_learning_fn(X: np.ndarray, Y: np.ndarray):
    """
    Handles both training and prediction for the XGBoost model.
    If train=True, X is training data and Y is target, returns trained model.
    If train=False, X is prediction data and model is trained model, returns predictions.
    """
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'verbosity': 0
    }
    
    # Training logic
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)
        
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    return bst # Return the trained XGBoost model

def xgboost_predict_fn(X: np.ndarray, model):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest) # Return predictions

ADNI = pd.read_csv('../data/ADNI.csv')
start_col = 'CTX_LH_BANKSSTS_SUVR_FTP_Inf_CER_GM_Non_PVC_v1'
end_col = 'rh_insula_thickness_Closest_AV1451_v1'

columns_to_check = ADNI.loc[:, start_col:].columns.tolist()

ADNI = ADNI.dropna(subset=columns_to_check)
ADNI = ADNI.dropna(subset=["PACC_Digit_LONI_Closest_AV1451_v1"])

Y = ADNI.iloc[:, 16].values
X_df = ADNI.iloc[:, 13:]
# Encode COMPOSITE column
X_df["Positivity_PET_ABETA_Closest_AV1451_v1"] = X_df["Positivity_PET_ABETA_Closest_AV1451_v1"].map({
    "COMPOSITE_N": -1.0,
    "COMPOSITE_P":  1.0
})

X = X_df.values

MODALITIES = {
        0: list(range(2)),
        1: list(range(6, 75)),   
        2: list(range(75, 142)) 
    }

# Initialize the ShapMML object
shapmml_model = ShapMML(
            x=X[:350],
            y=Y[:350],
            modalities=MODALITIES,
            learning_fn=xgboost_learning_fn,
            loss_fn=mse_loss_fn,
            predict_fn=xgboost_predict_fn,
            task_type="regression"
)

 # 1. Train base learners mu_S
shapmml_model.train()

# 2. Compute Shapley values ONCE
shapmml_model.marginal_calibrate()

print(shapmml_model.significance_table)

p = shapmml_model.p
num_modalities_list = np.arange(0, p+1)
test_mse_list = []
test_r2_list = []
counts_list = []

X_test = X[350:]
Y_true_test = Y[350:]

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

# Annotate counts for each q
for i, q in enumerate(num_modalities_list):
    counts_text = ', '.join(f'{k}:{v}' for k, v in counts_list[i].items())
    plt.text(q, test_mse_list[i]*1.01, counts_text, ha='center', fontsize=8, rotation=30)


plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("../output/msp_adni_mse.pdf")
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
plt.savefig("../output/msp_adni_r2.pdf")
plt.show()