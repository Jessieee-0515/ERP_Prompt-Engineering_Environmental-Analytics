import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load real training data (1990-1999)
# ------------------------------------------------------------

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

# 3. Governing ODE -- identical to the P3 run2 generation

def air2water_ode(Tw, t, Ta, alpha, beta):
    idx = min(int(t), len(Ta) - 1)
    dTw_dt = alpha * (Ta[idx] - Tw) + beta
    return dTw_dt

# 4. Numerical solution -- identical to the P3 run2 generation

def solve_ode(Ta, Tw_initial, alpha, beta):
    t = np.arange(len(Ta))
    Tw_sim = odeint(air2water_ode, Tw_initial, t, args=(Ta, alpha, beta))
    return Tw_sim.flatten()

# 5. Objective function -- identical to the P3 run2 generation

def objective_function(params, Ta, Tw_obs):
    alpha, beta = params
    Tw_initial = Tw_obs[0]
    Tw_sim = solve_ode(Ta, Tw_initial, alpha, beta)
    return mean_squared_error(Tw_obs, Tw_sim)

# 6. Calibrate model parameters -- ONLY on training data

def calibrate_parameters(Ta, Tw_obs):
    initial_guess = [0.1, 0.1]
    result = minimize(
        objective_function, initial_guess, args=(Ta, Tw_obs),
        method='L-BFGS-B', bounds=[(0, 1), (0, 1)]
    )
    return result.x
alpha, beta = calibrate_parameters(Ta_train, Tw_train)
print("Calibrated parameters (from 1990-1999 training data)")
print("------------------------------------------------------")
print(f"alpha = {alpha:.5f}")
print(f"beta  = {beta:.5f}")

# 7. Training-period fit

Tw_train_sim = solve_ode(Ta_train, Tw_train[0], alpha, beta)

train_rmse = np.sqrt(mean_squared_error(Tw_train, Tw_train_sim))
train_mae = mean_absolute_error(Tw_train, Tw_train_sim)
train_r2 = r2_score(Tw_train, Tw_train_sim)
train_nse = 1 - np.sum((Tw_train - Tw_train_sim) ** 2) / np.sum((Tw_train - np.mean(Tw_train)) ** 2)

# AIC (computed on training data only, k = number of calibrated parameters)
N_train = len(Tw_train)
k = 2  # alpha, beta
rss_train = np.sum((Tw_train - Tw_train_sim) ** 2)
train_aic = N_train * np.log(rss_train / N_train) + 2 * k

print("\nTraining Performance (1990-1999, in-sample)")
print("------------------------------------------------------")
print(f"RMSE = {train_rmse:.3f} °C")
print(f"MAE  = {train_mae:.3f} °C")
print(f"R²   = {train_r2:.3f}")
print(f"NSE  = {train_nse:.3f}")
print(f"AIC  = {train_aic:.2f}  (k={k})")

# 8. Validation-period forward simulation
#    only the forcing data changes.

Tw_val_sim = solve_ode(Ta_val, Tw_val[0], alpha, beta)
val_rmse = np.sqrt(mean_squared_error(Tw_val, Tw_val_sim))
val_mae = mean_absolute_error(Tw_val, Tw_val_sim)
val_r2 = r2_score(Tw_val, Tw_val_sim)
val_nse = 1 - np.sum((Tw_val - Tw_val_sim) ** 2) / np.sum((Tw_val - np.mean(Tw_val)) ** 2)

print("\nValidation Performance (2000-2004, out-of-sample)")
print("------------------------------------------------------")
print(f"RMSE = {val_rmse:.3f} °C")
print(f"MAE  = {val_mae:.3f} °C")
print(f"R²   = {val_r2:.3f}")
print(f"NSE  = {val_nse:.3f}")

# 9. Generalization gap

gap_rmse = val_rmse - train_rmse
gap_nse = val_nse - train_nse
print("\nGeneralization Gap")
print("------------------------------------------------------")
print(f"Validation RMSE - Training RMSE = {gap_rmse:.3f} °C")
print(f"Validation NSE  - Training NSE  = {gap_nse:.3f}")

# 10. Visualization -- both periods side by side

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(Tw_train, label='Observed', color='black', linewidth=1.2)
axes[0].plot(Tw_train_sim, label='Simulated', color='blue', linestyle='--', linewidth=1.2)
axes[0].set_title(f"P3 Fewshot Prompt Training (1990-1999)")
axes[0].set_ylabel("Water Temperature (°C)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(Tw_val, label='Observed', color='black', linewidth=1.2)
axes[1].plot(Tw_val_sim, label='Simulated', color='red', linestyle='--', linewidth=1.2)
axes[1].set_title(f"P3 Fewshot Prompt Validation (2000-2004)")
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
    f"P3 Few-shot Prompt — Validation Residuals (2000–2004)"
)

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()