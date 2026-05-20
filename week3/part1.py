# ── Part 1: Learning Curves & Polynomial Regression ──────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_california_housing
import pandas as pd


# 1. LOAD DATASET & USE TWO FEATURES FOR A RICH REGRESSION PROBLEM
raw  = fetch_california_housing()
df   = pd.DataFrame(raw.data, columns=raw.feature_names)


# Use MedInc (median income) as our single input — strong linear signal + noise
# Makes bias/variance tradeoff clearly visible across polynomial degrees
np.random.seed(42)
"""
 np.random(a,size,replace).
 len(df) --- number of rows in our dataset
 300     ----  number of samples to be taken from dataset
 replace if true means repitition allowed else not allowed.
 Matrix inversion has a computational complexity of roughly O(N^3). 
 Running this on all ~20,000 rows for high-degree polynomials would slow down our computer significantly
"""
sample_idx = np.random.choice(len(df), 300, replace=False)   # keep it manageable
X_raw = df['MedInc'].values[sample_idx]
y_raw = raw.target[sample_idx]


# Sort by X so regression curves plot cleanly
# Sorting the x-axis forces the plotter to draw the line smoothly from left to right.
order  = np.argsort(X_raw)
X_raw  = X_raw[order]
y_raw  = y_raw[order]


# 2. TRAIN / VALIDATION SPLIT  (70 / 30)
split    = int(0.7 * len(X_raw))
X_train, X_val = X_raw[:split], X_raw[split:]
y_train, y_val = y_raw[:split], y_raw[split:]

print(f"Dataset  : California Housing  (n={len(X_raw)}, 1 feature: MedInc)")
print(f"Training : {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples\n")


# 3. FROM-SCRATCH HELPERS  (no sklearn regression)
def make_poly_features(x, degree):
    """Build [1, x, x², …, x^degree] design matrix  (n × degree+1).
    This expression generates a matrix of polynomial features by stacking powers of x from 0 to the specified degree as separate columns.
    """
    return np.column_stack([x**d for d in range(degree + 1)])

def normal_equation(X, y):
    """
    Here  we calcualte inverse.This may be computitionally expensive if we have not done random sampling.
    Exact closed-form solution: θ = (XᵀX)⁻¹ Xᵀy
    Normal equation is used to get the optimal θ values in a single step although it is computationally expensive.
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def mse(y_true, y_pred): 
    # Mean squared error is used to find how much prediction is far away from actual value. 
    return np.mean((y_true - y_pred) ** 2)


def predict(X_raw, theta, degree):
    return make_poly_features(X_raw, degree) @ theta


# 4. FIT POLYNOMIAL MODELS  d ∈ {1, 2, 3, 4, 5}
DEGREES = [1, 2, 3, 4, 5]

thetas       = {}
train_errors = {}
val_errors   = {}

print(f"{'Degree':<8} {'Train MSE':>12} {'Val MSE':>12}  Diagnosis")
print("─" * 55)


# iterating over the DEGREES(1,2,3,4,5) as d to get polynomail features of degree 'd' and
# get the values of different thetas,train_errors & val_errors...
for d in DEGREES:
    X_tr_poly = make_poly_features(X_train, d)
    X_vl_poly = make_poly_features(X_val,   d)

    theta = normal_equation(X_tr_poly, y_train)
    tr_mse = mse(y_train, X_tr_poly @ theta)
    vl_mse = mse(y_val,   X_vl_poly @ theta)

    thetas[d]       = theta
    train_errors[d] = tr_mse
    val_errors[d]   = vl_mse

    if d == 1:
        diagnosis = "Underfit  (high bias)"
    elif d in [2, 3]:
        diagnosis = "Good fit"
    else:
        diagnosis = "Overfit   (high variance)"

    print(f"d={d}      {tr_mse:>12.4f} {vl_mse:>12.4f}  {diagnosis}")


# 5. PLOT A: REGRESSION CURVES FOR EACH DEGREE
X_line = np.linspace(X_raw.min(), X_raw.max(), 400)

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
fig.suptitle("Polynomial Regression Curves (d = 1 to 5)", fontsize=13, y=1.02)

colors = ['#e74c3c', '#e67e22', '#27ae60', '#2980b9', '#8e44ad']
labels = ['Underfit', 'Mild underfit', 'Good fit', 'Slight overfit', 'Overfit']

for ax, d, col, lbl in zip(axes, DEGREES, colors, labels):
    y_line = predict(X_line, thetas[d], d)

    ax.scatter(X_train, y_train, s=12, alpha=0.4, color='#95a5a6', label='Train')
    ax.scatter(X_val,   y_val,   s=12, alpha=0.4, color='#bdc3c7', marker='x', label='Val')
    ax.plot(X_line, y_line, color=col, linewidth=2.5, label=f'd={d}')

    ax.set_title(f"Degree {d}\n({lbl})", fontsize=10)
    ax.set_xlabel("MedInc (normalized)", fontsize=8)
    if d == 1:
        ax.set_ylabel("House Price ($100k)", fontsize=8)

    # Annotate MSE inside the plot
    ax.text(0.04, 0.95,
            f"Train MSE: {train_errors[d]:.3f}\nVal MSE:   {val_errors[d]:.3f}",
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('plot_A_regression_curves.png', dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLOT B: TRAIN/VAL ERROR vs DEGREE  (complexity curve)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(DEGREES, [train_errors[d] for d in DEGREES],
        'o-', color='#2980b9', linewidth=2, markersize=7, label='Training error')
ax.plot(DEGREES, [val_errors[d] for d in DEGREES],
        's--', color='#e74c3c', linewidth=2, markersize=7, label='Validation error')

ax.axvspan(0.8, 1.5, alpha=0.08, color='red',   label='Underfitting zone')
ax.axvspan(3.5, 5.2, alpha=0.08, color='purple', label='Overfitting zone')
ax.axvspan(1.5, 3.5, alpha=0.08, color='green',  label='Good fit zone')

ax.set_xlabel("Polynomial Degree", fontsize=11)
ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Training vs Validation Error by Model Complexity", fontsize=12)
ax.set_xticks(DEGREES)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot_B_complexity_curve.png', dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PLOT C: LEARNING CURVES  (error vs training set SIZE)
#    The question specifically asks for this — error as a function of training size
#    Done for d=1 (underfit), d=3 (good), d=5 (overfit)
# ═══════════════════════════════════════════════════════════════════════════════
SIZES       = np.linspace(10, len(X_train), 20, dtype=int)
PLOT_DEGREES = [1, 3, 5]
lc_colors   = {'train': '#2980b9', 'val': '#e74c3c'}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle("Learning Curves: Training Error & Validation Error vs Training Set Size",
             fontsize=13, y=1.02)

diagnoses = {1: "Degree 1 — Underfitting\n(both errors high, converge fast → high bias)",
             3: "Degree 3 — Good Fit\n(low train & val error, gap closes → balanced)",
             5: "Degree 5 — Overfitting\n(train low, val high, large gap → high variance)"}

for ax, d in zip(axes, PLOT_DEGREES):
    tr_curve, vl_curve = [], []

    for size in SIZES:
        Xtr_s = make_poly_features(X_train[:size], d)
        Xvl_s = make_poly_features(X_val,          d)

        th    = normal_equation(Xtr_s, y_train[:size])
        tr_curve.append(mse(y_train[:size], Xtr_s @ th))
        vl_curve.append(mse(y_val,          Xvl_s @ th))

    ax.plot(SIZES, tr_curve, color=lc_colors['train'], linewidth=2,
            label='Training error')
    ax.plot(SIZES, vl_curve, color=lc_colors['val'],   linewidth=2,
            linestyle='--', label='Validation error')

    ax.fill_between(SIZES, tr_curve, vl_curve,
                    alpha=0.12, color='purple', label='Bias-variance gap')

    ax.set_title(diagnoses[d], fontsize=9.5)
    ax.set_xlabel("Training set size", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot_C_learning_curves.png', dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. BIAS-VARIANCE EXPLANATION  (printed to console)
# ═══════════════════════════════════════════════════════════════════════════════
explanation = """
╔══════════════════════════════════════════════════════════════════════╗
║           BIAS–VARIANCE TRADEOFF OBSERVATIONS                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  UNDERFITTING  (d = 1)                                               ║
║  • Training error  : HIGH                                            ║
║  • Validation error: HIGH (similar to training)                      ║
║  • Gap             : SMALL                                           ║
║  • Cause           : Model is too simple (high bias). A straight     ║
║    line cannot capture the nonlinear housing price relationship.     ║
║  • Learning curve  : Both curves plateau early at high error.        ║
║    Adding more data does NOT help — the model is too constrained.    ║
║                                                                      ║
║  GOOD FIT  (d = 2–3)                                                 ║
║  • Training error  : LOW                                             ║
║  • Validation error: LOW and close to training error                 ║
║  • Gap             : SMALL                                           ║
║  • Cause           : Model complexity matches data complexity.       ║
║  • Learning curve  : Val error decreases and approaches train        ║
║    error as size grows — classic well-fitted behaviour.              ║
║                                                                      ║
║  OVERFITTING  (d = 4–5)                                              ║
║  • Training error  : VERY LOW (model memorises training data)        ║
║  • Validation error: HIGH (fails to generalise)                      ║
║  • Gap             : LARGE                                           ║
║  • Cause           : Model is too complex (high variance). It fits   ║
║    noise in the training set, not the true signal.                   ║
║  • Learning curve  : Large persistent gap between curves. Adding     ║
║    more data helps slowly, but gap remains without regularisation.   ║
║                                                                      ║
║  SUMMARY: Total Error = Bias² + Variance + Irreducible Noise        ║
║  • Low degree  → high bias²,   low variance  (underfit)             ║
║  • High degree → low bias²,    high variance (overfit)              ║
║  • Optimal degree balances both terms (d=2 or d=3 here)             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
print(explanation)