# ----- WEEK 2 Part 2: Normal Equation vs Gradient Descent -----
import numpy as np 
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pandas as pd

# ── 1. CREATE DATASET ──────────────────────────────────────────────────────────
np.random.seed(0)
n = 100
# 100 points between -3 and 3, inclusive, with some noise added to make it more realistic
# X is the input feature, y is the target variable with a cubic relationship to X plus some noise
X = np.linspace(-3, 3, n) 
y = 0.5*X**3 - X**2 + 2*X + 1 + np.random.normal(0, 2.5, n)

split    = int(0.7 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── 2. FEATURE MATRICES ────────────────────────────────────────────────────────
# Model 1 — degree 1 (linear), bias column prepended manually
X_train_m1 = np.c_[np.ones(len(X_train)), X_train]
X_test_m1  = np.c_[np.ones(len(X_test)),  X_test]

""" 
Model 2 — degree 3 (polynomial), bias column prepended manually
 Takes input X and expands it to third degree polynomial features dataset: [x, x², x³], then we prepend a bias column of 1s.
 include_bias=False: Normally, PolynomialFeatures automatically adds a column of entirely 1s to act as the bias term.
  However, since we are manually adding a bias column of 1s using np.c_, we set include_bias=False to prevent PolynomialFeatures from adding another redundant bias column.
  This way, we have full control over the bias term and avoid having two identical columns of 1s in our feature matrix.
  Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
  
"""
poly        = PolynomialFeatures(degree=3, include_bias=False) 
X_train_m2  = np.c_[np.ones(len(X_train)), poly.fit_transform(X_train.reshape(-1,1))] 
X_test_m2   = np.c_[np.ones(len(X_test)),  poly.transform(X_test.reshape(-1,1))]



# ── 3. ALGORITHMS ──────────────────────────────────────────────────────────────
"""
Normal Equation and Gradient Descent implementations for linear regression.
The normal equation provides a closed-form solution, while gradient descent is an iterative optimization algorithm.
Equating the loss finction's gradient to zero and solving for θ gives us the normal equation, which can be computed directly without iterations.
Gradient descent, on the other hand, starts with an initial guess for θ and iteratively updates
it in the direction of the steepest descent of the loss function, which requires tuning of learning rate and number of iterations.
Normal equation is computationally efficient for small datasets but can be slow for large datasets due to matrix inversion,
 while gradient descent can handle larger datasets but may require careful tuning to converge to the optimal solution.
"""
def normal_equation(X, y):
    """θ = (XᵀX)⁻¹ Xᵀy  — closed-form solution, no iterations needed."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y



def gradient_descent(X, y, lr=0.01, iterations=10000, seed=42):
    """Batch gradient descent with a fixed seed for reproducibility."""
    np.random.seed(seed)                         #  fixed seed → stable params
    m     = len(y)
    theta = np.random.randn(X.shape[1])


    """ 
    As lr=0.01 means our model is taking very small steps to move towards lowest error points.
     If we take small number(or just 1) iterations it will never reach to lowest loss point.
     so we take large iterations so that our model will take large number of small steps.
     """
    for _ in range(iterations):
        grad  = (2/m) * X.T @ (X @ theta - y)
        theta -= lr * grad
    return theta



# ── 4. EVALUATE & BUILD RESULTS ────────────────────────────────────────────────
def evaluate(name, X_train, y_train, X_test, y_test, method, lr=None, epochs=None):
    # Keeping the time just to take note of how much time gradient_descent is taking to update weights anf biasis.
    t0 = time.perf_counter()


    if method == "Normal Equation":
        theta = normal_equation(X_train, y_train)

    else:
        theta = gradient_descent(X_train, y_train, lr=lr, epochs=epochs)
    elapsed = time.perf_counter() - t0
    

    """
    Calculating mean square error for the train and test dataset.By calculating mean square error
    we just want to calculate how much(magnitude) our model differ from the actual value.
    """
    train_mse = mean_squared_error(y_train, X_train @ theta)
    test_mse  = mean_squared_error(y_test, X_test @ theta)

    #  add a meaningful Remarks string per row
    if method == "Normal Equation":
        remark = "Exact solution in one step; fast for small datasets"
    else:
        remark = f"Iterative; lr={lr}, epochs={epochs}; may need tuning"

    # return the calulated ouputs..
    # Read the next few lines of code below.It is simple.Assigning values to variables and calling functions.
    return {
        "Method":         f"{name} — {method}",
        "Train MSE":      round(train_mse, 4),
        "Test MSE":       round(test_mse,  4),
        "Time (s)":       round(elapsed,   6),
        "Remarks":        remark,           #  was completely missing
        "_theta":         theta,            # kept for parameter comparison & plots
        "_X_tr":          X_train,
        "_X_te":          X_test,
    }

rows = [
    evaluate("Model 1 (linear)",  X_train_m1, y_train, X_test_m1, y_test,
             "Normal Equation"),
    evaluate("Model 1 (linear)",  X_train_m1, y_train, X_test_m1, y_test,
             "Gradient Descent", lr=0.01,    epochs=5000),
    evaluate("Model 2 (cubic)",   X_train_m2, y_train, X_test_m2, y_test,
             "Normal Equation"),
    evaluate("Model 2 (cubic)",   X_train_m2, y_train, X_test_m2, y_test,
             "Gradient Descent", lr=0.0001,  epochs=50000),
]



# ── 5. COMPARISON TABLE ────────────────────────────────────────────────────────
display_cols = ["Method", "Train MSE", "Test MSE", "Time (s)", "Remarks"]
df_results   = pd.DataFrame(rows)[display_cols]

print("\n" + "="*90)
print("  PERFORMANCE COMPARISON TABLE")
print("="*90)
print(df_results.to_string(index=False))
print("="*90)

# ── 6. PARAMETER COMPARISON ( aligned side-by-side, not raw dumps) ────────
param_labels_m1 = ["θ₀ (bias)", "θ₁ (x)"]
param_labels_m2 = ["θ₀ (bias)", "θ₁ (x)", "θ₂ (x²)", "θ₃ (x³)"]




def print_param_table(label, labels, ne_theta, gd_theta):
    print(f"\n  {label}")
    print(f"  {'Parameter':<14} {'Normal Eq':>12} {'Grad Desc':>12} {'Difference':>12}")
    print(f"  {'-'*52}")
    for name, nv, gv in zip(labels, ne_theta, gd_theta):
        print(f"  {name:<14} {nv:>12.4f} {gv:>12.4f} {abs(nv-gv):>12.6f}")

print("\n" + "="*90)
print("  LEARNED PARAMETERS COMPARISON")
print("="*90)
print_param_table("Model 1 — Linear",
                  param_labels_m1, rows[0]["_theta"], rows[1]["_theta"])
print_param_table("Model 2 — Cubic Polynomial",
                  param_labels_m2, rows[2]["_theta"], rows[3]["_theta"])



# ── 7. PLOTS ( visualisation was completely missing) ──────────────────────
X_line = np.linspace(-3, 3, 300)


# Prediction curves for plotting
def predict_m1(theta, x): return np.c_[np.ones(len(x)), x] @ theta
def predict_m2(theta, x):
    Xp = poly.transform(x.reshape(-1, 1))
    return np.c_[np.ones(len(x)), Xp] @ theta

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Normal Equation vs Gradient Descent — Fitted Curves", fontsize=13)

titles  = ["Model 1 — Linear", "Model 2 — Cubic Polynomial"]
pred_fns = [predict_m1, predict_m2]
ne_thetas = [rows[0]["_theta"], rows[2]["_theta"]]
gd_thetas = [rows[1]["_theta"], rows[3]["_theta"]]



for ax, title, fn, ne_t, gd_t in zip(axes, titles, pred_fns, ne_thetas, gd_thetas):
    ax.scatter(X_train, y_train, s=18, alpha=0.5, color='#888', label='Train data')
    ax.scatter(X_test,  y_test,  s=18, alpha=0.5, color='#aaa', marker='x', label='Test data')
    ax.plot(X_line, fn(ne_t, X_line), color="#307aca", linewidth=2.5,
            label=f'Normal Eq  (test MSE={mean_squared_error(y_test, fn(ne_t, X_test)):.2f})')
    ax.plot(X_line, fn(gd_t, X_line), color='#c0392b', linewidth=2,
            linestyle='--',
            label=f'Grad Desc  (test MSE={mean_squared_error(y_test, fn(gd_t, X_test)):.2f})')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('normal_eq_vs_gd.png', dpi=150)
plt.show()
print("\nPlot saved as normal_eq_vs_gd.png")