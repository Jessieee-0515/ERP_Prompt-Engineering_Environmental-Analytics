import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# 3. Governing ODE -- identical to the P3 complex run2

def air2water_ode(Tw, t, Ta, params):
    K, Teq_offset = params
    Teq = Ta[int(t)] + Teq_offset
    dTw_dt = K * (Teq - Tw)
    return dTw_dt

#Solver for the ODE

def solve_ode(Ta, Tw0, params, t):
    Tw_sim = odeint(air2water_ode, Tw0, t, args=(Ta, params))
    return Tw_sim.flatten()

# Objective function for optimization

def objective_function(params, Ta, Tw_obs, t):
    Tw_sim = solve_ode(Ta, Tw_obs[0], params, t)
    rmse = np.sqrt(np.mean((Tw_sim - Tw_obs) ** 2))
    return rmse

# Calibration of model parameters -- identical to the original

def calibrate_model(Ta_obs, Tw_obs):
    t = np.arange(len(Ta_obs))
    initial_params = [0.1, 0.0]  # Initial guesses for K and Teq_offset
    result = minimize(objective_function, initial_params, args=(Ta_obs, Tw_obs, t), method='L-BFGS-B')
    return result.x

# Main function to run the model -- identical to the original,

def run_model(Ta_obs, Tw_obs):
    # Calibrate model parameters
    calibrated_params = calibrate_model(Ta_obs, Tw_obs)

    # Solve the ODE with calibrated parameters
    t = np.arange(len(Ta_obs))
    Tw_sim = solve_ode(Ta_obs, Tw_obs[0], calibrated_params, t)

    # Calculate goodness-of-fit metric
    rmse = np.sqrt(np.mean((Tw_sim - Tw_obs) ** 2))
    print(f"Calibrated Parameters: {calibrated_params}")
    print(f"RMSE: {rmse}")

    return calibrated_params  # <-- added so the params can be reused below

# 4. Run calibration on the TRAINING data only

print("Training (1990-1999)")
print("------------------------------------------------------")
calibrated_params = run_model(Ta_train, Tw_train)
K_opt, Teq_offset_opt = calibrated_params


# 5. Full training-period metrics (RMSE above already printed
#    by run_model(); the rest are added here for the study's
#    standard reporting set: MAE, R^2, NSE, AIC)

t_train = np.arange(len(Ta_train))
Tw_train_sim = solve_ode(Ta_train, Tw_train[0], calibrated_params, t_train)

train_mae = mean_absolute_error(Tw_train, Tw_train_sim)
train_r2 = r2_score(Tw_train, Tw_train_sim)
train_nse = 1 - np.sum((Tw_train - Tw_train_sim) ** 2) / np.sum((Tw_train - np.mean(Tw_train)) ** 2)

N_train = len(Tw_train)
k = 2  # K, Teq_offset
rss_train = np.sum((Tw_train - Tw_train_sim) ** 2)
train_aic = N_train * np.log(rss_train / N_train) + 2 * k

print(f"MAE  = {train_mae:.3f} °C")
print(f"R²   = {train_r2:.3f}")
print(f"NSE  = {train_nse:.3f}")
print(f"AIC  = {train_aic:.2f}  (k={k} parameters)")

# 6. Validation-period forward simulation (NO re-optimization)

t_val = np.arange(len(Ta_val))
Tw_val_sim = solve_ode(Ta_val, Tw_val[0], calibrated_params, t_val)

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

# 7. Generalization gap

train_rmse = np.sqrt(mean_squared_error(Tw_train, Tw_train_sim))
gap_rmse = val_rmse - train_rmse
gap_nse = val_nse - train_nse

print("\nGeneralization Gap")
print("------------------------------------------------------")
print(f"Validation RMSE - Training RMSE = {gap_rmse:.3f} °C")
print(f"Validation NSE  - Training NSE  = {gap_nse:.3f}")

# 8. Visualization -- both periods side by side

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(Tw_train, label='Observed', color='black', linewidth=1.2)
axes[0].plot(Tw_train_sim, label='Simulated', color='blue', linestyle='--', linewidth=1.2)
axes[0].set_title(f"Training (1990-1999) -- RMSE = {train_rmse:.3f} °C, NSE = {train_nse:.3f}")
axes[0].set_ylabel("Water Temperature (°C)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(Tw_val, label='Observed', color='black', linewidth=1.2)
axes[1].plot(Tw_val_sim, label='Simulated', color='red', linestyle='--', linewidth=1.2)
axes[1].set_title(f"Validation (2000-2004) -- RMSE = {val_rmse:.3f} °C, NSE = {val_nse:.3f}")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Water Temperature (°C)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Residual Analysis -- Validation Period

residual_val = Tw_val - Tw_val_sim

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

fig, ax = plt.subplots(figsize=(12, 4.5))

ax.plot(
    np.arange(len(residual_val)),
    residual_val,
    color="blue",
    linewidth=0.8,
    label="Residual"
)
ax.axhline(0, color="black", linestyle="--", linewidth=1.0, label="Zero residual")

ax.set_xlabel("Day")
ax.set_ylabel("Residual (°C)")
ax.set_title("P3 Few-shot (Complex) Prompt — Validation Residuals (2000–2004)")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()