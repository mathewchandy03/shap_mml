import cvxpy as cp
from sklearn.metrics.pairwise import pairwise_kernels
from itertools import chain, combinations
import numpy as np
from typing import Callable
import math
from scipy.stats import t
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from typing import Literal

class ShapMML:
    def __init__(
            self, 
            x : np.ndarray,
            y : np.ndarray,
            modalities : dict,
            learning_fn : Callable,
            predict_fn : Callable,
            loss_fn : Callable,
            task_type : Literal["regression", "classification"],
            lambda1 : float = None,  # Regularization for Kernel component
            lambda2 : float = None,  # Regularization for Bias/Linear component
            alpha = 0.05,           # Target coverage (e.g., 0.05 for 95th percentile)
            split : float = 0.5
        ):
        self.shape = x.shape[1:]
        self.n = len(y)
        self.split = split
        self.m = int(split * self.n)
        self.n_cal = self.n - self.m
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.x_train = self.x[:self.m]
        self.x_calib = self.x[self.m:]
        self.y_train = self.y[:self.m]
        self.y_calib = self.y[self.m:]
        self.modalities = modalities
        self.p = len(modalities)
        modality_list = list(range(self.p))
        self.modality_power_set = list(chain.from_iterable(combinations(modality_list, j) for j in range(len(modality_list) + 1)))
        self.learning_fn = learning_fn
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.loss_fn = loss_fn
        self.h = {} # Will store the predictors
        self.predict_fn = predict_fn
        self.task_type = task_type
        self.output_dim = len(np.unique(self.y_train)) if self.task_type == 'classification' else 1


    def mask(self, S):
        mask = np.zeros(self.shape, dtype=np.float32)
        if S is not None:
            for j in S:
                for idx in self.modalities[j]:
                    mask[idx] = 1.0
        # mask = np.expand_dims(mask, axis=(0, -1)) # Adjust based on input shape requirements
        return mask

    def train(self):
        mu = dict()
        for S in self.modality_power_set:
            # Apply mask. Note: Ensure broadcasting works with x_train structure
            # Assuming x_train is (N, Features) and mask is (Features,)
            x_train_mask = self.x_train * self.mask(S)
            mu[S] = self.learning_fn(x_train_mask, self.y_train)
        self.mu = mu

    def t_p_value(self, shapley_values):
        """
        One-sided (lower) conformal inference for
        H0: phi_new <= 0 vs H1: phi_new > 0
        """
        import pandas as pd
        n_cal, p = shapley_values.shape
        alpha = self.alpha

        rows = []

        for j in range(p):
            phi_j = shapley_values[:, j]

            ell = int(np.ceil((n_cal + 1) * alpha / 2))
            ell = max(ell, 1)

            u = int(np.ceil((n_cal + 1) * (1 - alpha / 2)))
            u = min(u, n_cal)

            # Lower conformal bound
            L_j = np.sort(phi_j)[ell - 1]
            U_j = np.sort(phi_j)[u - 1]
            # Conformal p-value
            # p_value = (1 + np.sum(phi_j <= 0.0)) / (n_cal + 1)
            mean_j = np.mean(phi_j)
            std_j = np.std(phi_j, ddof=1)
            t_stat = mean_j / (std_j / np.sqrt(n_cal))
            if std_j < 1e-12:
                p_value = 1.0
            else:
                p_value = 1 - t.cdf(t_stat, df=n_cal - 1)  # one-sided test H0: mean <= 0
            rows.append({
                "modality": j,
                "lower_bound": L_j,
                "upper_bound": U_j,
                "p_value": p_value
               # "reject_H0_at_alpha": L_j > 0
            })

        return pd.DataFrame(rows)

    def marginal_calibrate(self):
        baseline = None
        utility = dict()
        
        # --- 1. Compute Utility (Loss) for every subset S ---
        baseline = self.loss_fn(self.y_calib, self.predict_fn(self.x_calib * self.mask(()), self.mu[()]
))
        for S in self.modality_power_set:
            if S == ():
                loss = baseline
            else:
                x_calib_mask = self.x_calib * self.mask(S)
                y_pred = self.predict_fn(x_calib_mask, self.mu[S])
                loss = self.loss_fn(self.y_calib, y_pred)

            assert loss.ndim == 1 and loss.shape[0] == self.n_cal
            
            utility[S] = baseline - loss
        
        
        p = self.p
        
        # --- 2. Compute Shapley Values ---
        fact = np.array([math.factorial(i) for i in range(p+1)], dtype=np.float32)
        shapley_values = np.zeros((self.n_cal, p), dtype=np.float32)
        
        for i in range(self.p):
            others = [j for j in range(p) if j != i]
            for r in range(len(others)+1):
                for subset in combinations(others, r):
                    S = tuple(sorted(subset))
                    S_i = tuple(sorted(subset + (i,)))
                    
                    weight = np.float32(fact[len(S)] * fact[p - len(S) - 1] / fact[p])
                    v_S = utility.get(S, 0)
                    v_Si = utility.get(S_i, 0)
                    shapley_values[:, i] += weight * (v_Si - v_S)

        self.shapley_values = shapley_values
        self.significance_table = self.t_p_value(shapley_values)

    def tune_lambdas(
        self,
        lambda1_grid,
        lambda2_grid,
        q,
        dim_reduce: {"pca", "svd"},
        n_folds=5,
        random_state=0,
        n_components = None
    ):
        if q == 0:
            return None, None   # in tune_lambdas

        X = self.x_calib
        Phi = self.shapley_values
        n = X.shape[0]
        self.dim_reduce = dim_reduce
        self.n_components = n_components

        kf = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )

        best_score = -np.inf
        best_lams = None

        for lam1 in lambda1_grid:
            for lam2 in lambda2_grid:

                fold_scores = []

                for train_idx, val_idx in kf.split(X):
                    X_train = X[train_idx]
                    Phi_train = Phi[train_idx]

                    X_val = X[val_idx]
                    Phi_val = Phi[val_idx]

                    # Fit h on K-1 folds
                    self.lambda1 = lam1
                    self.lambda2 = lam2

                    self.conditional_calibrate(
                        q=q,
                        dim_reduce = dim_reduce, 
                        X_calib=X_train,
                        Phi_calib=Phi_train,
                        n_components = n_components
                    )

                    # Predict h on held-out fold
                    cache_key = (
                        dim_reduce,
                        n_components,
                        id(X_train)
                    )

                    h_val = self.predict(X_val, cache_key)

                    # Shapley-mass score
                    score = 0.0
                    for i in range(len(X_val)):
                        pos_idx = np.where(h_val[i] > 0)[0]
                        if len(pos_idx) == 0:
                            continue

                        ranked = pos_idx[np.argsort(h_val[i, pos_idx])[::-1]]
                        selected = ranked[:q]   # possibly < q
                        score += Phi_val[i, selected].sum()

                    score /= len(X_val)
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)

                if avg_score > best_score:
                    best_score = avg_score
                    best_lams = (lam1, lam2)

        return best_lams

    def conditional_calibrate(self, q, dim_reduce: {"pca", "svd"}, X_calib = None, Phi_calib = None, n_components = None):
        # ---------------------------------------------------------
        # Semi-Parametric RKHS Quantile Regression
        # Model: h_j(x) = K(x, X_calib)^T * eta_j + x^T * beta_j
        # ---------------------------------------------------------
        if q == 0:
            self.q = 0
            self.selected_modalities_ = []
            return

        if X_calib is None:
            X_calib = self.x_calib
            n_cal = self.n_cal
        else:
            n_cal = X_calib.shape[0]
        if Phi_calib is None:
            Phi_calib = self.shapley_values
        if X_calib is not None and Phi_calib is None:
            raise ValueError("Phi_calib must be provided when X_calib is provided")

        # Prepare Data
        X_gamma = X_calib.reshape(n_cal, -1)

        if not hasattr(self, "_cond_cache"):
            self._cond_cache = {}

        cache_key = (
            dim_reduce,
            n_components,
            id(X_calib)
            )
        
        self._last_cache_key = cache_key


        
        if cache_key not in self._cond_cache:
            # Center the data
            mu = np.mean(X_gamma, axis=0)
            X_gamma_centered = X_gamma - mu

            if dim_reduce == "pca":
                pca = PCA(n_components=n_components, whiten=False, random_state=0)
                V_prime_calib = pca.fit_transform(X_gamma_centered)
                # transform = {"type": "pca", "pca": pca}
                Q = None
            elif dim_reduce == "svd":
                # Compute Covariance Matrix Σ
                # Sigma = np.cov(X_gamma_centered, rowvar=False) 
                
                # Compute SVD of Σ: Σ = Q^T Λ Q (or eigendecomposition)
                # We need Q (orthogonal matrix) for the transformation.
                # np.linalg.svd returns U, s, Vh where Sigma = U @ diag(s) @ Vh
                # If Sigma is covariance, U=V=Q (orthogonal matrix of eigenvectors)
                # Using SVD on the centered data directly:
                # U, s, Vh = np.linalg.svd(X_phi_centered, full_matrices=False)
                # This is more numerically stable and gives us the principal components transformation matrix Vh^T = Q
                _, _, Vh = np.linalg.svd(X_gamma_centered, full_matrices=False)
                Q = Vh.T # Orthogonal matrix Q (contains eigenvectors)
                pca = None
                # Transform features to the decorrelated space V'
                V_prime_calib = X_gamma_centered @ Q 

                if n_components is not None:
                    Q = Q[:, :n_components]
                    V_prime_calib = V_prime_calib[:, :n_components]

            else:
                raise ValueError("dim_reduce must be 'pca' or 'svd'")
        
            # 2. Kernel Matrix (RBF) and CVXPY Variables
            
            # Dimension of the transformed feature space (V')
            dim_v_prime = V_prime_calib.shape[1]
            
            gamma = 1.0 / (dim_v_prime if dim_v_prime > 0 else 1)
            K = pairwise_kernels(V_prime_calib, V_prime_calib, metric='rbf', gamma=gamma)
            K += 1e-6 * np.eye(n_cal) # Jitter
            
            self._cond_cache[cache_key] = {
            "mu": mu,
            "Q": Q,
            "V_prime_calib": V_prime_calib,
            "K": K,
            "pca": pca,
            "gamma": gamma
            }
        
        cache = self._cond_cache[cache_key]
        mu = cache["mu"]
        Q = cache["Q"]
        V_prime_calib = cache["V_prime_calib"]
        K = cache["K"]
        p = self.p
        pca = cache["pca"]
        gamma = cache["gamma"]

        dim_v_prime = V_prime_calib.shape[1]

        # Variables: Beta now has dimension dim_v_prime
        # -------------------------------------------------------
# Warm-started CVXPY solves (DROP-IN REPLACEMENT)
# -------------------------------------------------------

# storage for warm starts
        if not hasattr(self, "_warm_start"):
            self._warm_start = {}

        warm_key = (cache_key, self.lambda1, self.lambda2)

        eta_opt = np.zeros((n_cal, p))
        beta_opt = np.zeros((dim_v_prime, p))

        for j in range(p):
            eta_j = cp.Variable(n_cal)
            beta_j = cp.Variable(dim_v_prime)

            pred_j = K @ eta_j + V_prime_calib @ beta_j
            u_j = Phi_calib[:, j] - pred_j

            tau = 1 - self.alpha / (2 * q)
            pinball = 0.5 * cp.abs(u_j) + (tau - 0.5) * u_j
            loss = cp.sum(pinball) / n_cal

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-4 * np.eye(n_cal))

            reg = self.lambda1 * cp.sum_squares(L.T @ eta_j) \
                + self.lambda2 * cp.sum_squares(beta_j)

            prob = cp.Problem(cp.Minimize(loss + reg))

            # ---- WARM START (THIS IS THE ONLY NEW PART) ----
            if warm_key in self._warm_start:
                eta_j.value = self._warm_start[warm_key][0][:, j]
                beta_j.value = self._warm_start[warm_key][1][:, j]


            try:
                prob.solve(
                    solver=cp.ECOS,
                    warm_start=True,
                    abstol=1e-6,
                    reltol=1e-6,
                    feastol=1e-6,
                    verbose=False
                )
            except cp.error.SolverError:
                prob.solve(
                    solver=cp.OSQP,
                    warm_start=True,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    verbose=False
    )

            eta_opt[:, j] = eta_j.value
            beta_opt[:, j] = beta_j.value

        # save solution for next call
        self._warm_start[warm_key] = (eta_opt.copy(), beta_opt.copy())




        
        
        def create_predictor(eta_col, beta_col, V_prime_calib_ref):
            def predictor(x_test, Q_ref, mu_ref, pca_ref, gamma_ref):
                # Ensure x_test is flat for transformation
                n_test = x_test.shape[0]
                x_test_flat = x_test.reshape(n_test, -1)
                X_phi_centered_test = x_test_flat - mu_ref
                
                if Q_ref is None:
                    V_prime_test = pca_ref.transform(X_phi_centered_test)
                else:
                    V_prime_test = X_phi_centered_test @ Q_ref
                # Apply Transformation: X -> Phi(X) -> V'
                
                
                # RKHS Part: K(V'_test, V'_calib)^T * eta
                K_test = pairwise_kernels(V_prime_test, V_prime_calib_ref, metric='rbf', gamma=gamma_ref)
                rkhs_term = K_test @ eta_col
                
                # Linear Part: V'_test @ beta
                linear_term = V_prime_test @ beta_col
                
                return rkhs_term + linear_term
            return predictor

        h = {}
        for j in range(self.p):
            # The predictor needs to take x_test and the transformation params (Q, mu)
            h[j] = create_predictor(eta_opt[:, j], beta_opt[:, j], V_prime_calib)
            
        self.h = h
        self.q = q

    def predict(self, x_test, cache_key = None):
        if cache_key is None:
            if not hasattr(self, "_last_cache_key"):
                raise RuntimeError("No calibrated model available")
            cache_key = self._last_cache_key
        # Uses the stored transformation parameters Q and mu
        Q_ref = self._cond_cache[cache_key]['Q']
        mu_ref = self._cond_cache[cache_key]['mu']
        pca_ref = self._cond_cache[cache_key]['pca']
        gamma_ref = self._cond_cache[cache_key]['gamma']
        
        n_test = x_test.shape[0]
        preds = np.zeros((n_test, self.p))
        
        for i in range(self.p):
            # Pass the transformation parameters Q and mu to the predictor function
            preds[:, i] = self.h[i](x_test, Q_ref, mu_ref, pca_ref, gamma_ref)
            
        return preds
    
    def select_and_mask(self, x_test: np.ndarray) -> tuple[np.ndarray, list]:
        """
        For each observation in x_test, selects the q modalities with the highest
        predicted conditional quantile (h) and applies the corresponding mask.

        Args:
            x_test: The test data array (n_test, ...).
            q_modalities: The number of top modalities to select.

        Returns:
            A tuple: (masked_x_test, selected_modalities_list)
            - masked_x_test: The input masked by the selected modalities.
            - selected_modalities_list: A list of the selected modality indices (S) for each test observation.
        """
        
        n_test = x_test.shape[0]
        
        # 1. Predict the conditional quantile h_j for all modalities (h returns (n_test, p))
        # This uses the method implemented in the previous step.
        h_scores = self.predict(x_test) 
        
        # Initialize output storage
        masked_x_test = np.zeros_like(x_test, dtype=np.float32)
        selected_modalities_list = []

        for i in range(n_test):
            # 2. Select the top q modalities based on h_scores
            
            # Get the scores for observation i
            scores_i = h_scores[i, :]
            pos_idx = np.where(scores_i > 0)[0]

            if len(pos_idx) == 0:
                selected_modalities_list.append(())
                continue

            ranked = pos_idx[np.argsort(scores_i[pos_idx])[::-1]]
            min_q = min(self.q, len(ranked))
            selected_indices = ranked[:min_q]
            
            # S is the set of modality indices for this observation
            S = tuple(sorted(selected_indices.tolist()))
            selected_modalities_list.append(S)
            
            # 3. Apply the mask for the selected set S
            # The mask generation needs the context of the modalities dictionary.
            # We need to reshape the mask to apply it to x_test[i]
            modality_mask = self.mask(S)
            
            # Apply mask and store the result
            masked_x_test[i] = x_test[i] * modality_mask

        return masked_x_test, selected_modalities_list
    
    def predict_optimal_modalities(self, x_test: np.ndarray) -> np.ndarray:
        """
        Selects the q optimal modalities based on h-scores for each observation 
        and generates the final prediction using the learned predictor mu_S.

        Args:
            x_test: The test data array (n_test, ...).
            q_modalities: The number of top modalities to select.

        Returns:
            np.ndarray: The final prediction Y_hat for each test observation.
        """
        if self.q == 0:
            n_test = x_test.shape[0]

            if self.task_type == "classification":
                # Predict the marginal class distribution
                class_probs = np.bincount(
                    self.y_train.astype(int),
                    minlength=self.output_dim
                ).astype(float)
                class_probs /= class_probs.sum()

                y_pred = np.tile(class_probs, (n_test, 1))
            else:
                # Regression: marginal mean
                y_pred = np.full(n_test, self.y_train.mean())

            return y_pred, [tuple()] * n_test


        # 1. Select the top modalities and mask the input
        masked_x_test, selected_modalities_list = self.select_and_mask(x_test)
                
        
        # Initialize predictions (assuming output shape matches y_train)
        # NOTE: You must adjust the shape initialization if y is not a simple vector.
        n_test = x_test.shape[0]
        if self.task_type == 'classification':
            y_pred = np.zeros((n_test, self.output_dim), dtype=np.float32)
        else:
            y_pred = np.zeros((n_test,), dtype=np.float32)  # scalar per sample


        
        # Group test observations by the selected modality set S
        # This avoids re-running prediction for the same trained model mu_S
        groups = {}
        for i, S in enumerate(selected_modalities_list):
            if S not in groups:
                groups[S] = []
            groups[S].append(i)
            
        # 2. Iterate through groups and make predictions
        for S, indices in groups.items():
            model_mu = self.mu[S]
            x_group = masked_x_test[indices]

            # --- FIX: enforce batch dimension ---
            if x_group.ndim == x_test.ndim - 1:
                x_group = x_group[None, ...]
            # -----------------------------------

            y_group_pred = self.predict_fn(x_group, model_mu)
            
            if self.task_type == 'classification':
                y_pred[indices, :] = y_group_pred
            else:
                y_pred[indices] = y_group_pred

 
        return y_pred, selected_modalities_list