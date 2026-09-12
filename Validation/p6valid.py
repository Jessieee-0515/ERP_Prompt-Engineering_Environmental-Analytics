import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time as timer

program_start = timer.time()

# 1. Load real training data (1990-1999)

train_df = pd.read_csv(
    "../output/45004_1990_1999.csv",
    parse_dates=["Datetime"]
)
train_df = train_df[["Datetime", "ATMP", "WTMP"]]

daily_train = (
    train_df
    .set_index("Datetime")
    .resample("D")
    .mean()
)
daily_train = daily_train.dropna(subset=["ATMP", "WTMP"])

Ta_train = daily_train["ATMP"].to_numpy(dtype=float)
Tw_train = daily_train["WTMP"].to_numpy(dtype=float)

# 2. Load real validation data (2000-2004)

val_df = pd.read_csv(
    "../output/45004_2000_2004.csv",
    parse_dates=["Datetime"]
)
val_df = val_df[["Datetime", "ATMP", "WTMP"]]

daily_val = (
    val_df
    .set_index("Datetime")
    .resample("D")
    .mean()
)
daily_val = daily_val.dropna(subset=["ATMP", "WTMP"])

Ta_val = daily_val["ATMP"].to_numpy(dtype=float)
Tw_val = daily_val["WTMP"].to_numpy(dtype=float)

# 3. Governing ODE

def air2water_ode(t, Tw, Ta, params):
    p1, p2, p3, p4, p5, p6, p7, p8 = params
    Tr = 4.0
    tyr = 365.0

    if Tw >= Tr:
        delta = np.exp((Tr - Tw) / p6)
    else:
        delta = np.exp((Tw - Tr) / p7) + np.exp(-Tw / p8)

    dTw_dt = (1 / delta) * (p1 * np.cos(2 * np.pi * (t / tyr - p2)) + p3 + p4 * (Ta - Tw) + p5 * Tw)
    return dTw_dt

# 4. Forward simulation helper

def simulate(Ta_obs, Tw_obs, params):
    print("simulating")
    sol = solve_ivp(
        fun=lambda t, Tw: air2water_ode(t, Tw, Ta_obs[int(t)], params),
        t_span=(0, len(Ta_obs) - 1),
        y0=[Tw_obs[0]],
        t_eval=np.arange(len(Ta_obs))
    )
    return sol.y[0]

# 5. Objective function

def objective_function(params, Ta_obs, Tw_obs):

    print("objective called")
    print(params)

    try:
        Tw_sim = simulate(Ta_obs, Tw_obs, params)

        if not np.all(np.isfinite(Tw_sim)):
            return 1e10

        nse = (
            1
            - np.sum((Tw_obs - Tw_sim) ** 2)
            / np.sum((Tw_obs - np.mean(Tw_obs)) ** 2)
        )

        if not np.isfinite(nse):
            return 1e10

        return -nse

    except Exception as e:
        print("Simulation failed:", e)
        return 1e10

# 6. Parameter bounds

bounds = [
    (0.00, 0.33),  # p1
    (0.00, 1.00),  # p2
    (-0.12, 0.28), # p3
    (0.00, 0.01),  # p4
    (-0.02, 0.00), # p5
    (0, 15),       # p6
    (0, 15),       # p7
    (0, 0.5)       # p8
]

# 7. Calibrate

generation = 0
de_start = timer.time()



result = differential_evolution(
    func=objective_function,
    bounds=bounds,
    args=(Ta_train, Tw_train),
    strategy='best1bin',
    maxiter=20,
    popsize=5,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
)

best_params = result.x

print("Calibrated parameters (from 1990-1999 training data)")
print("------------------------------------------------------")
param_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
for name, value in zip(param_names, best_params):
    print(f"{name} = {value:.5f}")

# 8. Training-period fit (in-sample)

Tw_train_sim = simulate(Ta_train, Tw_train, best_params)

train_nse = 1 - np.sum((Tw_train - Tw_train_sim) ** 2) / np.sum((Tw_train - np.mean(Tw_train)) ** 2)
train_rmse = np.sqrt(mean_squared_error(Tw_train, Tw_train_sim))
train_mae = mean_absolute_error(Tw_train, Tw_train_sim)
train_r2 = r2_score(Tw_train, Tw_train_sim)

N_train = len(Tw_train)
k = 8  # <-- number of calibrated parameters in THIS model; change per file

rss_train = np.sum((Tw_train - Tw_train_sim) ** 2)
train_aic = N_train * np.log(rss_train / N_train) + 2 * k

print("\nTraining Performance (1990-1999, in-sample)")
print("------------------------------------------------------")
print(f"NSE  = {train_nse:.3f}")
print(f"RMSE = {train_rmse:.3f} °C")
print(f"MAE  = {train_mae:.3f} °C")
print(f"R²   = {train_r2:.3f}")
print(f"AIC  = {train_aic:.2f}  (k={k})")

# 9. Validation-period

Tw_val_sim = simulate(Ta_val, Tw_val, best_params)

val_nse = 1 - np.sum((Tw_val - Tw_val_sim) ** 2) / np.sum((Tw_val - np.mean(Tw_val)) ** 2)
val_rmse = np.sqrt(mean_squared_error(Tw_val, Tw_val_sim))
val_mae = mean_absolute_error(Tw_val, Tw_val_sim)
val_r2 = r2_score(Tw_val, Tw_val_sim)

print("\nValidation Performance (2000-2004, out-of-sample)")
print("------------------------------------------------------")
print(f"NSE  = {val_nse:.3f}")
print(f"RMSE = {val_rmse:.3f} °C")
print(f"MAE  = {val_mae:.3f} °C")
print(f"R²   = {val_r2:.3f}")

# 10. Generalization gap

gap_rmse = val_rmse - train_rmse
gap_nse = val_nse - train_nse

print("\nGeneralization Gap")
print("------------------------------------------------------")
print(f"Validation RMSE - Training RMSE = {gap_rmse:.3f} °C")
print(f"Validation NSE  - Training NSE  = {gap_nse:.3f}")

# 11. Visualization -- both periods side by side

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(Tw_train, label='Observed', color='black', linewidth=1.2)
axes[0].plot(Tw_train_sim, label='Simulated', color='blue', linestyle='--', linewidth=1.2)
axes[0].set_title(f"P6 Rag Prompt Training (1990-1999)")
axes[0].set_ylabel("Water Temperature (°C)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(Tw_val, label='Observed', color='black', linewidth=1.2)
axes[1].plot(Tw_val_sim, label='Simulated', color='red', linestyle='--', linewidth=1.2)
axes[1].set_title(f"P6 RAG Prompt Validation (2000-2004)")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Water Temperature (°C)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 11. Residual Analysis -- Validation Period

# Residual definition:
# Residual = Observed - Simulated
residual_val = Tw_val - Tw_val_sim

# Residual statistics

residual_mean = np.mean(residual_val)
residual_std = np.std(residual_val)
residual_rmse = np.sqrt(np.mean(residual_val ** 2))
residual_mae = np.mean(np.abs(residual_val))

print("\nResidual Analysis (Validation 2000-2004)")
print("------------------------------------------------------")
print(f"Mean residual = {residual_mean:.3f} °C")
print(f"Std residual  = {residual_std:.3f} °C")
print(f"Residual RMSE = {residual_rmse:.3f} °C")
print(f"Residual MAE  = {residual_mae:.3f} °C")

# Residual time-series plot

fig, ax = plt.subplots(figsize=(12, 4.5))

ax.plot(
    np.arange(len(residual_val)),
    residual_val,
    color="blue",
    linewidth=0.8,
    label="Residual"
)

ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="Zero residual"
)

ax.set_xlabel("Day")
ax.set_ylabel("Residual (°C)")

ax.set_title(
    f"P6 RAG — Validation Residuals (2000–2004)"
)

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()