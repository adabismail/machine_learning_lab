# ── Part 3: Cross-Validation to Select Optimal Regularisation λ ──────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd


# 1. LOAD DATA  (same dataset & feature as Parts 1 & 2 for consistency)
raw = fetch_california_housing()
df  = pd.DataFrame(raw.data, columns=raw.feature_names)


np.random.seed(42)
# Choosing random numbers without repetition from 0 to length of dataset.
idx   = np.random.choice(len(df), 300, replace=False)
X_raw = df['MedInc'].values[idx]
y_raw = raw.target[idx]


# ── 70 / 30 split (train used for CV, test kept completely aside) ──────────
split   = int(0.7 * len(X_raw))
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train,      y_test     = y_raw[:split], y_raw[split:]

print(f"Total samples : 300")
print(f"Training set  : {len(y_train)} samples  ← used for 5-fold CV")
print(f"Test set      : {len(y_test)}  samples  ← held out until final evaluation\n")



# 2. FROM-SCRATCH HELPERS

def make_poly_features(x, degree=3):
    """Build design matrix: columns = [1, x, x², x³]"""
    return np.column_stack([x**d for d in range(degree + 1)])



def ridge_normal_equation(X, y, lam):
    """
    Ridge regression closed-form solution:
        θ = (XᵀX + λI)⁻¹ Xᵀy

    Adding λI to XᵀX does two things:
      1. Makes the matrix invertible even when features are correlated.
      2. Shrinks θ values toward zero → penalises model complexity → reduces overfitting.
    Note: We do NOT regularise the bias term (first column), so we zero out
          the (0,0) entry of the identity matrix.
    """
    n_features = X.shape[1]
    I          = np.eye(n_features)
    I[0, 0]    = 0                          # do not penalise the bias term
    return np.linalg.inv(X.T @ X + lam * I) @ X.T @ y




def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


"""
 3. 5-FOLD CROSS-VALIDATION
    For every λ:
      • Split training data into 5 equal folds
      • Train on 4 folds, validate on the remaining 1 fold
      • Repeat 5 times (each fold gets a turn as the validation fold)
      • Average the 5 validation errors → that is the CV error for this λ
"""
LAMBDAS  = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
K        = 5          # number of folds
DEGREE   = 3          # polynomial degree (good fit from Part 1)


# Build the full polynomial feature matrix for the training set ONCE
X_train_poly = make_poly_features(X_train_raw, DEGREE)

n_train      = len(y_train)
fold_size    = n_train // K      # how many samples in each fold

print(f"Running {K}-fold cross-validation with degree-{DEGREE} polynomial...")
print(f"{'λ':<10} {'Fold 1':>8} {'Fold 2':>8} {'Fold 3':>8} {'Fold 4':>8} {'Fold 5':>8} {'Mean CV MSE':>12}")
print("─" * 72)

cv_mean_errors = []   # one average CV error per λ

for lam in LAMBDAS:
    fold_errors = []

    for fold in range(K):
        # ── Identify validation indices for this fold ──────────────────────
        val_start = fold * fold_size
        val_end   = val_start + fold_size           # last fold may be slightly larger
        if fold == K - 1:
            val_end = n_train                       # include any remainder in last fold

        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start),
                                    np.arange(val_end, n_train)])

        # ── Split features and labels ──────────────────────────────────────
        X_fold_train = X_train_poly[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val   = X_train_poly[val_idx]
        y_fold_val   = y_train[val_idx]

        # ── Train with ridge regression ────────────────────────────────────
        theta      = ridge_normal_equation(X_fold_train, y_fold_train, lam)
        fold_error = mse(y_fold_val, X_fold_val @ theta)
        fold_errors.append(fold_error)

    mean_cv_error = np.mean(fold_errors)
    cv_mean_errors.append(mean_cv_error)

    # Print one row per λ showing all fold errors + mean
    fold_str = "  ".join(f"{e:8.4f}" for e in fold_errors)
    print(f"{lam:<10}  {fold_str}  {mean_cv_error:12.4f}")



# 4. SELECT BEST λ
best_idx = np.argmin(cv_mean_errors)
best_lam = LAMBDAS[best_idx]

print(f"\n{'='*72}")
print(f"  Best λ found : {best_lam}  (lowest CV MSE = {cv_mean_errors[best_idx]:.4f})")
print(f"{'='*72}\n")





# 5. RETRAIN ON ENTIRE TRAINING SET  using best λ
#    Now that we know the best λ from CV, we use ALL 210 training samples
#    (no fold splitting) to get the best possible final model.

theta_final = ridge_normal_equation(X_train_poly, y_train, best_lam)

# ── Final evaluation on the TEST set (never seen during CV) ───────────────
X_test_poly  = make_poly_features(X_test_raw, DEGREE)
test_mse     = mse(y_test, X_test_poly @ theta_final)
train_mse    = mse(y_train, X_train_poly @ theta_final)

print("FINAL MODEL RESULTS")
print(f"  Regularisation λ   : {best_lam}")
print(f"  Polynomial degree  : {DEGREE}")
print(f"  Train MSE          : {train_mse:.4f}")
print(f"  Test  MSE          : {test_mse:.4f}")
print(f"  Learned parameters : {np.round(theta_final, 4)}")




# 6. PLOT 1 — Validation Error vs λ  (log scale on x-axis)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("5-Fold Cross-Validation — Regularisation Parameter Selection", fontsize=13)

ax = axes[0]
ax.semilogx(LAMBDAS, cv_mean_errors,
            'o-', color='#2980b9', linewidth=2.5, markersize=8, label='Mean CV MSE')

# Highlight the best λ
ax.semilogx(best_lam, cv_mean_errors[best_idx],
            '*', color='#e74c3c', markersize=18, label=f'Best λ = {best_lam}', zorder=5)

# Show individual fold errors as faint lines behind the mean
all_fold_data = []
for lam in LAMBDAS:
    fold_errors = []
    for fold in range(K):
        val_start = fold * fold_size
        val_end   = val_start + fold_size
        if fold == K - 1:
            val_end = n_train
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_train)])
        theta     = ridge_normal_equation(X_train_poly[train_idx], y_train[train_idx], lam)
        fold_errors.append(mse(y_train[val_idx], X_train_poly[val_idx] @ theta))
    all_fold_data.append(fold_errors)

for fold in range(K):
    fold_curve = [all_fold_data[i][fold] for i in range(len(LAMBDAS))]
    ax.semilogx(LAMBDAS, fold_curve,
                '--', color='#aab7b8', linewidth=0.9, alpha=0.6,
                label=f'Fold {fold+1}' if fold == 0 else '_')

ax.set_xlabel("Regularisation parameter λ  (log scale)", fontsize=11)
ax.set_ylabel("Validation MSE", fontsize=11)
ax.set_title("Validation Error vs λ", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
ax.annotate(f"Best λ = {best_lam}",
            xy=(best_lam, cv_mean_errors[best_idx]),
            xytext=(best_lam * 3, cv_mean_errors[best_idx] + 0.01),
            fontsize=9, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2))



# 7. PLOT 2 — Final model fitted curve on train + test data

ax2 = axes[1]
X_line     = np.linspace(X_raw.min(), X_raw.max(), 400)
y_line     = make_poly_features(X_line, DEGREE) @ theta_final

ax2.scatter(X_train_raw, y_train, s=18, alpha=0.5, color='#2980b9', label='Train data')
ax2.scatter(X_test_raw,  y_test,  s=18, alpha=0.5, color='#e74c3c',
            marker='x', linewidths=1.2, label='Test data')
ax2.plot(X_line, y_line, color='#1a252f', linewidth=2.5,
         label=f'Final model (λ={best_lam})')

ax2.set_xlabel("MedInc", fontsize=11)
ax2.set_ylabel("House Price ($100k)", fontsize=11)
ax2.set_title(f"Final Ridge Model  |  Test MSE = {test_mse:.4f}", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cross_validation_results.png', dpi=150)
plt.show()
print("\nPlot saved as cross_validation_results.png")