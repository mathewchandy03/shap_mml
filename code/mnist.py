import numpy as np
from shap_mml import ShapMML
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import label_binarize



# -------------------------------
# Experiment
# -------------------------------

(X, Y), (X_test, Y_true_test) = mnist.load_data()

X = X.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X = X[..., None]
X_test = X_test[..., None]


grid_size = 28

# Side lengths per block (sums to 28)
block_sizes = [9, 9, 10]

# Compute cumulative boundaries
row_starts = np.cumsum([0] + block_sizes[:-1])
col_starts = np.cumsum([0] + block_sizes[:-1])

MODALITIES = [
    [
        (r, c)
        for r in range(row_start, row_start + row_size)
        for c in range(col_start, col_start + col_size)
    ]
    for row_start, row_size in zip(row_starts, block_sizes)
    for col_start, col_size in zip(col_starts, block_sizes)
]

length = X.shape[1]


# -------------------------------
# Learning & Prediction Functions
# -------------------------------



def build_model():
    model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def nn_learning_fn(X, Y):
    model = build_model()
    model.fit(X, Y,
              epochs=1,
              batch_size=16,
              validation_split=0.1,
              verbose=1)
    return model


def nn_predict_fn(X, model):
    return model.predict(X, verbose=0)

def ce(y_true, y_pred):
    """
    y_true: shape (n,) with integer labels 0,...,C-1
    y_pred: shape (n, C) predicted probabilities
    Returns scalar average log-loss over all samples
    """
    y_onehot = label_binarize(y_true, classes=np.arange(y_pred.shape[1]))
    # Per-sample log-loss
    per_sample_loss = -np.sum(y_onehot * np.log(np.clip(y_pred, 1e-15, 1-1e-15)), axis=1)
    return np.mean(per_sample_loss)

shapmml_model = ShapMML(
    x=X,
    y=Y,
    modalities=MODALITIES,
    learning_fn=nn_learning_fn,
    predict_fn=nn_predict_fn,
    loss_fn = lambda y, yhat: -np.sum(
    label_binarize(y, classes=np.arange(yhat.shape[1])) * np.log(np.clip(yhat, 1e-15, 1-1e-15)),
    axis=1
),
    task_type="classification",
    split = 0.99
)

shapmml_model.train()
pred_example = shapmml_model.predict_fn(X_test, shapmml_model.mu[()])

print("Trained", flush = True)

shapmml_model.marginal_calibrate()

print("Marginally Calibrated", flush = True)

print(shapmml_model.significance_table, flush = True)

p = shapmml_model.p
num_modalities_list = np.arange(0, p+1)
test_ce_list = []
test_acc_list = []
counts_list = []


for q in num_modalities_list:
    # 3. Tune lambdas for fixed q
    # lam1_opt, lam2_opt = shapmml_model.tune_lambdas(
    #     lambda1_grid=[1e-3, 1e-2],
    #     lambda2_grid=[1e-3, 1e-2],
    #     q=q,
    #     dim_reduce = "pca",
    #     n_components = 2,
    #     n_folds = 2
    # )

    # print("Tuning for " + str(q) + " done", flush = True)

    # # 4. Refit conditional quantile model on full calibration set
    shapmml_model.lambda1 = 1e-2
    shapmml_model.lambda2 = 1e-2
    shapmml_model.conditional_calibrate(q, dim_reduce = "pca", n_components = 2)

    print("Conditional Calibration for " + str(q) + " done", flush = True)

    # 5. Select modalities and predict
    
    Y_pred_optimal, selected_modalities_list = shapmml_model.predict_optimal_modalities(X_test)

    print("Prediction for " + str(q) + " done", flush = True)

    # Test CE
    ce_val = ce(Y_true_test, Y_pred_optimal)
    test_ce_list.append(ce_val)

    # Test Accuracy
    y_hat = np.argmax(Y_pred_optimal, axis=1)
    acc_val = 100 * np.mean(Y_true_test == y_hat)
    test_acc_list.append(acc_val)
    
    # Count selected modality sets
    selected_counts = Counter(selected_modalities_list)
    counts_list.append(selected_counts)
    
    print(str(q) + " done", flush = True)
# Plot
plt.figure(figsize=(7,5))
plt.plot(num_modalities_list, test_ce_list, marker='o', linestyle='-', color='b', label='Test CE')
plt.xticks(num_modalities_list)
plt.xlabel("Max Number of Modalities (q)")
plt.ylabel("Test CE")
plt.title("Model Selection Path")

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("../output/msp_mnist_ce.pdf")
plt.show()



plt.figure(figsize=(7,5))
plt.plot(num_modalities_list, test_acc_list, marker='o', linestyle='-', color='b', label='Test Accuracy (%)')
plt.xticks(num_modalities_list)
plt.xlabel("Max Number of Modalities (q)")
plt.ylabel("Test Accuracy")
plt.title("Model Selection Path")

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("../output/msp_mnist_acc.pdf")
plt.show()