import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

"""
np.linspace(start,end,n) gives an array of size 'n'...evenly separated values form start - end.
we try to create a dataset and make our algorithm to find the best fit model.
Since in reality there are some noise due to human errors like typos etc.
So to mimic the real world data we will add some noise by generating 'n' random numbers between 0 & 3.
By adding this noise to y_true, we get y, our final target variable.
 X_raw and y are what you will actually feed into your model, challenging it to ignore the noise and find the hidden cubic curve.
"""
np.random.seed(42)
n = 80
X_raw = np.linspace(-3, 3, n)
y_true = 0.5 * X_raw**3 - X_raw**2 + 2 * X_raw + 1
y = y_true + np.random.normal(0, 3.0, n)



# Split 70 / 30
split = int(0.7 * n)
X_train_raw, X_val_raw = X_raw[:split], X_raw[split:]
y_train,      y_val     = y[:split],      y[split:]



# ── Build degree-7 polynomial feature matrix (captures enough complexity) ──
def poly_features(x, degree=7):
    """Returns [1, x, x^2, ..., x^degree] design matrix (N x (degree+1))."""
    return np.column_stack([x**d for d in range(degree + 1)])

"""
We will not always know underlying pattern of the data.In this case we know that our data 
has degree-3 underlying pattern.We take a highly complex model of degree-7 to understand how
we can put a leash (limit) to our model to prevent our model going crazy (overfitting).
For that we will use 'lambda' a hyperparameter.It will be disscussed below.
"""
DEGREE = 7
X_train = poly_features(X_train_raw, DEGREE)   # (56, 8)
X_val   = poly_features(X_val_raw,   DEGREE)   # (24, 8)
X_plot_raw = np.linspace(-3, 3, 300)
X_plot  = poly_features(X_plot_raw,  DEGREE)

# ── Feature normalisation (zero-mean, unit-variance per column, skip bias) ──
mean_ = X_train[:, 1:].mean(axis=0)
std_  = X_train[:, 1:].std(axis=0) + 1e-8

"""
Normalising the values using mean and standard deviation using the above calulated values.
The reason to do noramlisation is disscussed earlier in week2_part2..
"""
def normalise(X):
    Xn = X.copy().astype(float)
    Xn[:, 1:] = (Xn[:, 1:] - mean_) / std_
    return Xn

X_train_n = normalise(X_train)
X_val_n   = normalise(X_val)
X_plot_n  = normalise(X_plot)


"""
We use(Lambda)—also known as the Regularization Parameter—to act as a leash on the model,
 preventing it from going crazy. It alters the loss function so the model now has to minimize two 
 things:Loss = Mean Squared Error + lambda * Complexity Penalty
 We always move in a way which gives us less Loss.Which means we have to reduce both MSE and 
 Complexity of model.Keeping lambda low ~Zero means we will allow model to overfit.
 While keeping it high will mean that our model will underfit by giving less priority to mean square
 error (by assigning large lambda values means high change in model complexity -->> 
 high changes in loss.Means reducing model complexity will reduce the loss greatly as priority 
 (mathematically) given to MSE is low.).So we have to provide a Moderate (0-1) value to the lambda.
"""
lambdas = [0, 0.01, 0.1, 1, 10]
colors  = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#c77dff']

# 2. GRADIENT DESCENT IMPLEMENTATIONS


# Function to calculate mean square loss.
def mse_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)


# ── Ridge Regression (L2) ────────────────────────────────────────────────────
def ridge_gradient_descent(X, y, lam, lr=0.01, n_iter=3000):
    """
    Minimises: (1/N) * ||y - Xw||^2  +  λ * ||w||^2
    Gradient : (2/N) * X^T (Xw - y)  +  2λ * w   (bias w[0] not penalised)
    Ridge regression shrinks the model's weights smoothly by penalizing their squared values.
    N is the number of rows (samples) and p is the number of columns
    """
    N, p = X.shape
    w = np.zeros(p)  # initailizing the weights by zero.
    history = []

    for _ in range(n_iter):
        y_pred = X @ w
        # Gradient calculating fromulla
        grad = (2 / N) * (X.T @ ( y_pred - y))
        """ L2 penalty — do NOT regularise the bias term (index 0) because it doesn't increase the
           the complexity of the model.It shifts the whole curve Up and Down (it is bias).
           2*lam*w is the derivative of  λ*||w||^2
        """        
        penalty = 2 * lam * w
        penalty[0] = 0.0
        # We update the weights by subtracting the gradient and the penalty, scaled by the learning rate (lr).
        w -= lr * (grad + penalty)
        history.append(mse_loss(y, y_pred) + lam * np.sum(w[1:]**2))

    return w, history



# ── Lasso Regression (L1) via Subgradient Descent ────────────────────────────
def lasso_gradient_descent(X, y, lam, lr=0.005, n_iter=5000):
    """
    Minimises: (1/N) * ||y - Xw||^2  +  λ * ||w||_1
    Gradient : (2/N) * X^T (Xw - y)  +  λ * sign(w)  (bias not penalised)
    Uses a decaying learning rate for better convergence.
    """
    N, p = X.shape
    w = np.zeros(p)
    history = []

    for i in range(n_iter):
        lr_i = lr / (1 + 0.001 * i)          # decay
        y_pred = X @ w
        # Gradient calculating formulla
        grad = (2 / N) * (X.T @  (y_pred - y))
        # L1 subgradient — do NOT regularise bias
        penalty = lam * np.sign(w)
        penalty[0] = 0.0
        w -= lr_i * (grad + penalty)
        history.append(mse_loss(y, X @ w) + lam * np.sum(np.abs(w[1:])))

    return w, history



# 3. TRAIN ALL MODELS
ridge_results = {}
lasso_results = {}

print("=" * 72)
print(f"{'MODEL':<8} {'λ':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'||w|| (L2)':>12}")
print("=" * 72)

for lam in lambdas:
    # Ridge
    w_r, hist_r = ridge_gradient_descent(X_train_n, y_train, lam)
    tr_r  = mse_loss(y_train, X_train_n @ w_r)
    val_r = mse_loss(y_val,   X_val_n   @ w_r)
    ridge_results[lam] = dict(w=w_r, history=hist_r, train=tr_r, val=val_r)
    print(f"{'Ridge':<8} {lam:>6}  {tr_r:>12.4f}  {val_r:>12.4f}  {np.linalg.norm(w_r[1:]):>12.4f}")

    # Lasso
    w_l, hist_l = lasso_gradient_descent(X_train_n, y_train, lam)
    tr_l  = mse_loss(y_train, X_train_n @ w_l)
    val_l = mse_loss(y_val,   X_val_n   @ w_l)
    lasso_results[lam] = dict(w=w_l, history=hist_l, train=tr_l, val=val_l)
    print(f"{'Lasso':<8} {lam:>6}  {tr_l:>12.4f}  {val_l:>12.4f}  {np.linalg.norm(w_l[1:]):>12.4f}")
    print("-" * 72)



# 4. FIGURE 1 — Ridge Regression Curves
fig, axes = plt.subplots(1, 5, figsize=(23, 5))
fig.patch.set_facecolor('#0d0d1a')

for i, (lam, col) in enumerate(zip(lambdas, colors)):
    ax = axes[i]
    ax.set_facecolor('#13132b')
    res = ridge_results[lam]
    y_pred_plot = X_plot_n @ res['w']

    ax.scatter(X_train_raw, y_train, color='#ffffff', alpha=0.5, s=18, zorder=2, label='Train')
    ax.scatter(X_val_raw,   y_val,   color='#88aaff', alpha=0.6, s=18, zorder=2, label='Val')
    ax.plot(X_plot_raw, y_pred_plot, color=col, lw=2.8, zorder=3)
    ax.plot(X_plot_raw, y_true[:1][0]*0 + np.interp(X_plot_raw, X_raw,
            0.5*X_raw**3 - X_raw**2 + 2*X_raw + 1),
            color='white', lw=1, ls='--', alpha=0.3, label='True fn')

    ax.set_title(f'λ = {lam}', color=col, fontsize=13, fontweight='bold')
    ax.set_xlabel('x', color='#aaaacc', fontsize=9)
    if i == 0:
        ax.set_ylabel('y', color='#aaaacc', fontsize=10)
        ax.legend(fontsize=7, framealpha=0.15, labelcolor='white', facecolor='#1a1a3a')
    ax.tick_params(colors='#666688', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#2a2a4a')
    ax.set_ylim(-25, 25)

    info = (f"Train: {res['train']:.2f}\n"
            f"Val:   {res['val']:.2f}\n"
            f"||w||: {np.linalg.norm(res['w'][1:]):.2f}")
    ax.text(0.04, 0.97, info, transform=ax.transAxes, color='#ccccee',
            fontsize=7.5, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a', alpha=0.7))

fig.suptitle('Ridge Regression (L2)  ·  Polynomial Degree 7  ·  Varying λ',
             color='white', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('fig4_ridge_curves.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print("\nSaved: fig4_ridge_curves.png")




# 5. FIGURE 2 — Lasso Regression Curves
fig, axes = plt.subplots(1, 5, figsize=(23, 5))
fig.patch.set_facecolor('#0d0d1a')

for i, (lam, col) in enumerate(zip(lambdas, colors)):
    ax = axes[i]
    ax.set_facecolor('#13132b')
    res = lasso_results[lam]
    y_pred_plot = X_plot_n @ res['w']

    ax.scatter(X_train_raw, y_train, color='#ffffff', alpha=0.5, s=18, zorder=2, label='Train')
    ax.scatter(X_val_raw,   y_val,   color='#88aaff', alpha=0.6, s=18, zorder=2, label='Val')
    ax.plot(X_plot_raw, y_pred_plot, color=col, lw=2.8, zorder=3)
    ax.plot(X_plot_raw, np.interp(X_plot_raw, X_raw,
            0.5*X_raw**3 - X_raw**2 + 2*X_raw + 1),
            color='white', lw=1, ls='--', alpha=0.3, label='True fn')

    ax.set_title(f'λ = {lam}', color=col, fontsize=13, fontweight='bold')
    ax.set_xlabel('x', color='#aaaacc', fontsize=9)
    if i == 0:
        ax.set_ylabel('y', color='#aaaacc', fontsize=10)
        ax.legend(fontsize=7, framealpha=0.15, labelcolor='white', facecolor='#1a1a3a')
    ax.tick_params(colors='#666688', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#2a2a4a')
    ax.set_ylim(-25, 25)

    zeros = np.sum(np.abs(res['w'][1:]) < 0.01)
    info = (f"Train: {res['train']:.2f}\n"
            f"Val:   {res['val']:.2f}\n"
            f"Zeros: {zeros}/{DEGREE}")
    ax.text(0.04, 0.97, info, transform=ax.transAxes, color='#ccccee',
            fontsize=7.5, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a', alpha=0.7))

fig.suptitle('Lasso Regression (L1)  ·  Polynomial Degree 7  ·  Varying λ',
             color='white', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('fig5_lasso_curves.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print("Saved: fig5_lasso_curves.png")




# 6. FIGURE 3 — Coefficient Magnitudes (Ridge vs Lasso)
fig, axes = plt.subplots(2, 5, figsize=(23, 8))
fig.patch.set_facecolor('#0d0d1a')
feature_names = [f'x^{d}' for d in range(DEGREE + 1)]

for col_i, (lam, col) in enumerate(zip(lambdas, colors)):
    for row_i, (label, results) in enumerate([('Ridge (L2)', ridge_results),
                                               ('Lasso (L1)', lasso_results)]):
        ax = axes[row_i][col_i]
        ax.set_facecolor('#13132b')
        w = results[lam]['w']
        bars = ax.bar(range(DEGREE + 1), w, color=col, alpha=0.8, edgecolor='#2a2a4a')
        ax.axhline(0, color='white', lw=0.6, alpha=0.4)
        ax.set_xticks(range(DEGREE + 1))
        ax.set_xticklabels(feature_names, rotation=45, fontsize=7, color='#aaaacc')
        ax.tick_params(colors='#666688', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#2a2a4a')
        if col_i == 0:
            ax.set_ylabel(f'{label}\nCoefficient', color='#aaaacc', fontsize=8)
        ax.set_title(f'λ = {lam}', color=col, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.10, color='white')

fig.suptitle('Coefficient Magnitudes: Ridge vs Lasso across λ values',
             color='white', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig6_coefficients.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print("Saved: fig6_coefficients.png")





# 7. FIGURE 4 — Loss Convergence Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d0d1a')

for ax, (title, results) in zip(axes, [('Ridge (L2)', ridge_results),
                                        ('Lasso (L1)', lasso_results)]):
    ax.set_facecolor('#13132b')
    for lam, col in zip(lambdas, colors):
        hist = results[lam]['history']
        ax.plot(hist, color=col, lw=2, label=f'λ={lam}')
    ax.set_xlabel('Iteration', color='#aaaacc', fontsize=11)
    ax.set_ylabel('Regularised Loss', color='#aaaacc', fontsize=11)
    ax.set_title(f'{title} — Loss Convergence', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#888899')
    for sp in ax.spines.values():
        sp.set_color('#2a2a4a')
    ax.legend(framealpha=0.2, labelcolor='white', facecolor='#1a1a3a', fontsize=10)
    ax.grid(True, alpha=0.10, color='white')
    ax.set_yscale('log')

fig.suptitle('Gradient Descent Convergence', color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig7_convergence.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print("Saved: fig7_convergence.png")





# 8. FIGURE 5 — Ridge vs Lasso Validation Error Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d0d1a')

for ax, (title, results) in zip(axes, [('Ridge (L2)', ridge_results),
                                        ('Lasso (L1)', lasso_results)]):
    ax.set_facecolor('#13132b')
    tr_vals  = [results[l]['train'] for l in lambdas]
    val_vals = [results[l]['val']   for l in lambdas]
    lam_labels = [str(l) for l in lambdas]

    ax.plot(lam_labels, tr_vals,  'o-', color='#6bcb77', lw=2.5, ms=9, label='Train MSE')
    ax.plot(lam_labels, val_vals, 's-', color='#ff6b6b', lw=2.5, ms=9, label='Val MSE')
    ax.set_xlabel('λ (Regularization Strength)', color='#aaaacc', fontsize=11)
    ax.set_ylabel('MSE', color='#aaaacc', fontsize=11)
    ax.set_title(f'{title} — Error vs λ', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#888899')
    for sp in ax.spines.values():
        sp.set_color('#2a2a4a')
    ax.legend(framealpha=0.2, labelcolor='white', facecolor='#1a1a3a', fontsize=10)
    ax.grid(True, alpha=0.12, color='white')

fig.suptitle('Training vs Validation MSE: Ridge vs Lasso',
             color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig8_ridge_vs_lasso.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print("Saved: fig8_ridge_vs_lasso.png")





# 9. FINAL PARAMETER TABLE
print("\n" + "=" * 72)
print("FINAL PARAMETER VALUES")
print("=" * 72)
header = f"{'Feature':<8}" + "".join([f"{'R λ='+str(l):>12}{'L λ='+str(l):>12}" for l in lambdas])
print(f"{'Feature':<8}", end="")
for l in lambdas:
    print(f"  R(λ={l})   L(λ={l})", end="")
print()
print("-" * 72)
for d in range(DEGREE + 1):
    print(f"{'x^'+str(d):<8}", end="")
    for l in lambdas:
        print(f"  {ridge_results[l]['w'][d]:>7.3f}  {lasso_results[l]['w'][d]:>7.3f}", end="")
    print()
print("=" * 72)