# -*- coding: utf-8 -*-
"""
Simulation script: ADAPTIVE Model Predictive Control (AMPC)
Optimized version: prices and PV are read directly from the theoretical optimum.
"""
import os
import pandas as pd
import numpy as np
import pulp
import time
import matplotlib.pyplot as plt

# =============================================================================
# 0. OUTPUT HELPER (PNG + PDF into two separate folders)
# =============================================================================
PNG_DIR = "figures_PNG"
PDF_DIR = "figures_PDF"
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)


def save_figure(fig, name, dpi=300):
    fig.savefig(os.path.join(PNG_DIR, name + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(PDF_DIR, name + ".pdf"), bbox_inches="tight")
    print(f">>> Saved: {name}.png / {name}.pdf")


print(">>> Starting ADAPTIVE MPC simulation (advanced closed-loop)...")
start_time = time.time()

# =============================================================================
# 1. SIMPLIFIED AND ALIGNED DATA LOADING
# =============================================================================
try:
    # 1. Real consumption
    real_consumption_df = pd.read_excel("CER_Carico_Orario_Aggregato_k.xlsx")
    real_consumption = real_consumption_df["Carico_kWh"].values[:8760]

    # 2. Predicted consumption (using the file aligned to 2025)
    pa_df = pd.read_excel("CER_Carico_PA.xlsx")
    lighting_df = pd.read_excel("CER_Carico_Illuminazione.xlsx")
    industry_df = pd.read_excel("CER_Carico_Industria.xlsx")
    residential_pred_df = pd.read_excel("CER_Carico_Residenziale_PREDETTO_25.xlsx")

    predicted_consumption = (pa_df["Carico_kWh"].values[:8760] +
                             lighting_df["Carico_kWh"].values[:8760] +
                             industry_df["Carico_kWh"].values[:8760] +
                             residential_pred_df["Carico_kWh"].values[:8760])

    # 3. Data from the theoretical optimum (PV, prices, tariffs)
    optimum_df = pd.read_excel("Risultati_Ottimizzazione_StandAlone_1xBESS.xlsx",
                               sheet_name="Dati Orari").iloc[:8760]
    pv_production = optimum_df["Produzione_FV_kWh"].values
    zonal_price = optimum_df["Pz_Eur_MWh"].values
    hourly_incentive_tariff = optimum_df["Tariffa_Incentivo_Oraria_Eur_MWh"].values

    if "Timestamp" in optimum_df.columns:
        timestamp_index = pd.to_datetime(optimum_df["Timestamp"])
    else:
        timestamp_index = pd.date_range(start="2025-01-01 00:00", periods=8760, freq="h")

    # 4. Open-loop plan
    open_plan_df = pd.read_excel("Piano_Operativo_BESS_1x_PREVISIONALE.xlsx")
    open_charge = open_plan_df["Carica_Batteria_Pianificata_kWh"].values[:8760]
    open_discharge = open_plan_df["Scarica_Batteria_Pianificata_kWh"].values[:8760]

except Exception as e:
    print(f"DATA LOADING ERROR: {e}")
    exit()

# Technical parameters
arera_fee = 10.57
n_BESS = 1
E_nom = 760 * n_BESS
P_max = 380 * n_BESS
eta_BESS = 0.95
eta_cycle = eta_BESS ** 0.5
SoC_min = 0.10
SoC_max = 1.00
SoC_initial = 0.5
E_min = E_nom * SoC_min
E_max = E_nom * SoC_max
n_hours = 8760
mpc_horizon = 24

# Fictitious economic value assigned to the energy left in the battery at the end of the day
terminal_energy_value = 100

# =============================================================================
# 2. OPEN-LOOP RECONSTRUCTION (for figures and comparison)
# =============================================================================
open_shared_energy = np.zeros(n_hours)
open_grid_injection = np.zeros(n_hours)
for h in range(n_hours):
    available_energy = max(0, (pv_production[h] - open_charge[h]) + open_discharge[h])
    open_shared_energy[h] = min(real_consumption[h], available_energy)
    open_grid_injection[h] = available_energy - open_shared_energy[h]

open_hourly_revenue = (open_shared_energy / 1000) * (hourly_incentive_tariff + arera_fee) + \
                      (open_grid_injection / 1000) * zonal_price

# =============================================================================
# 3. ADAPTIVE MPC SIMULATION
# =============================================================================
print(f">>> Starting ADAPTIVE MPC loop. Solving {n_hours} MILP problems...")

mpc_shared_energy = np.zeros(n_hours)
mpc_grid_injection = np.zeros(n_hours)
real_SoC = np.zeros(n_hours + 1)
real_SoC[0] = SoC_initial * E_nom

solver = pulp.PULP_CBC_CMD(msg=False)

for h in range(n_hours):
    if h % 1460 == 0:
        print(f"    - Processing... {(h / n_hours) * 100:.0f}% completed")

    H = min(mpc_horizon, n_hours - h)
    horizon = range(H)

    # --- FORECAST UPDATING (autoregressive filter) ---
    corrected_consumption_horizon = np.zeros(H)
    if h > 0:
        recent_error = real_consumption[h - 1] - predicted_consumption[h - 1]
    else:
        recent_error = 0

    for k in horizon:
        decay_weight = 0.6 ** k  # the error loses weight as we look further ahead
        correction = recent_error * decay_weight
        corrected_consumption_horizon[k] = max(0, predicted_consumption[h + k] + correction)
    # -------------------------------------------------

    prob = pulp.LpProblem("AMPC_Step", pulp.LpMaximize)

    E = pulp.LpVariable.dicts("E", horizon, lowBound=E_min, upBound=E_max)
    Pc = pulp.LpVariable.dicts("Pc", horizon, lowBound=0, upBound=P_max)
    Pd = pulp.LpVariable.dicts("Pd", horizon, lowBound=0, upBound=P_max)
    Pifv = pulp.LpVariable.dicts("Pifv", horizon, lowBound=0)
    Pibess = pulp.LpVariable.dicts("Pibess", horizon, lowBound=0)
    Ec = pulp.LpVariable.dicts("Ec", horizon, lowBound=0)
    is_c = pulp.LpVariable.dicts("d", horizon, cat=pulp.LpBinary)

    # OBJECTIVE: revenue over the H hours + terminal value of the energy left in the battery
    horizon_revenue = pulp.lpSum(
        (Ec[k] / 1000) * (float(hourly_incentive_tariff[h + k]) + arera_fee) +
        ((Pifv[k] + Pibess[k]) / 1000) * float(zonal_price[h + k]) for k in horizon)
    terminal_value = (E[H - 1] / 1000) * terminal_energy_value
    prob += horizon_revenue + terminal_value

    for k in horizon:
        prob += float(pv_production[h + k]) == Pc[k] + Pifv[k]
        prob += Pd[k] == Pibess[k]
        prob += Ec[k] <= corrected_consumption_horizon[k]  # use the CORRECTED estimates
        prob += Ec[k] <= Pifv[k] + Pibess[k]
        prob += Pc[k] <= P_max * is_c[k]
        prob += Pd[k] <= P_max * (1 - is_c[k])

        if k == 0:
            prob += E[k] == real_SoC[h] + Pc[k] * eta_cycle - Pd[k] / eta_cycle
        else:
            prob += E[k] == E[k - 1] + Pc[k] * eta_cycle - Pd[k] / eta_cycle

    prob.solve(solver)

    if pulp.LpStatus[prob.status] == "Optimal":
        cmd_Pc = Pc[0].varValue
        cmd_Pd = Pd[0].varValue
    else:
        cmd_Pc = 0
        cmd_Pd = 0

    # REAL EXECUTION of hour 0
    available_energy = max(0, (pv_production[h] - cmd_Pc) + cmd_Pd)
    mpc_shared_energy[h] = min(real_consumption[h], available_energy)
    mpc_grid_injection[h] = available_energy - mpc_shared_energy[h]

    real_SoC[h + 1] = max(E_min, min(E_max, real_SoC[h] + cmd_Pc * eta_cycle - cmd_Pd / eta_cycle))

mpc_hourly_revenue = (mpc_shared_energy / 1000) * (hourly_incentive_tariff + arera_fee) + \
                     (mpc_grid_injection / 1000) * zonal_price

# =============================================================================
# 4. FINAL RESULTS AND PRINTOUT
# =============================================================================
perfect_hourly_revenue = (optimum_df["Ricavo_Incentivo_Orario_Eur"] +
                          optimum_df["Ricavo_ARERA_Orario_Eur"] +
                          optimum_df["Ricavo_RID_Orario_Eur"])
perfect_revenue = perfect_hourly_revenue.sum()

open_total_revenue = np.sum(open_hourly_revenue)
mpc_total_revenue = np.sum(mpc_hourly_revenue)

print("\n" + "=" * 60)
print(" FINAL RESULTS: ADAPTIVE MPC vs OPEN-LOOP")
print("=" * 60)
print(f"1. Theoretical optimum (Perfect Foresight): EUR {perfect_revenue:,.2f}")
print(f"2. ADAPTIVE MPC revenue (with correction):  EUR {mpc_total_revenue:,.2f}")
print(f"3. Open-loop revenue (rigid plan):          EUR {open_total_revenue:,.2f}")
print("-" * 60)
recovery = mpc_total_revenue - open_total_revenue
open_loss = perfect_revenue - open_total_revenue
print(f">>> Open-loop loss: EUR {open_loss:,.2f} ({(open_loss / perfect_revenue) * 100:.2f}%)")
print(f">>> The Adaptive MPC RECOVERED: EUR {recovery:,.2f}")
print("=" * 60)

# =============================================================================
# 5. FIGURES
# =============================================================================
print(">>> Generating figures...")

plot_df = pd.DataFrame({
    "Sequential_Hours": range(n_hours),
    "Timestamp": timestamp_index,
    "Perfect_Cum": perfect_hourly_revenue.cumsum(),
    "Open_Cum": pd.Series(open_hourly_revenue).cumsum(),
    "MPC_Cum": pd.Series(mpc_hourly_revenue).cumsum(),
    "Shared_Perfect": optimum_df["Energia_Condivisa_kWh"],
    "Shared_Open": open_shared_energy,
    "Shared_MPC": mpc_shared_energy,
})

# --- Figure 1: Scissor plot (X axis = sequential numbers) ---
fig1 = plt.figure(figsize=(10, 6))
plt.plot(plot_df["Sequential_Hours"], plot_df["Perfect_Cum"],
         label="Theoretical optimum (Perfect)", color="green", linewidth=2.5)
plt.plot(plot_df["Sequential_Hours"], plot_df["MPC_Cum"],
         label="Closed-Loop (Adaptive MPC)", color="blue", linestyle="--", linewidth=2)
plt.plot(plot_df["Sequential_Hours"], plot_df["Open_Cum"],
         label="Open-Loop (rigid plan)", color="red", linestyle="-.", linewidth=1.5)
plt.title("Cumulative REC revenue (Forecast vs MPC vs Reality)")
plt.ylabel("Cumulative revenue [EUR]")
plt.xlabel("Hour of the year (0 - 8759)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
save_figure(fig1, "revenue_comparison_3scenarios")
plt.close(fig1)

# --- Figure 2: Critical week (with Timestamp) ---
plot_df["Month"] = plot_df["Timestamp"].dt.month
plot_df["Day_Of_Year"] = plot_df["Timestamp"].dt.dayofyear
worst_month = 8  # fixed to August, consistent with the other figures

worst_month_df = plot_df[plot_df["Month"] == worst_month]
# Worst day for the open-loop
worst_day = (worst_month_df["Shared_Perfect"] - worst_month_df["Shared_Open"]).groupby(
    worst_month_df["Day_Of_Year"]).sum().idxmax()

worst_day_index = plot_df[plot_df["Day_Of_Year"] == worst_day].index[0]
plot_start = max(0, worst_day_index - (24 * 3))
plot_end = min(n_hours, plot_start + (24 * 7))
week_df = plot_df.iloc[plot_start:plot_end]

fig2 = plt.figure(figsize=(12, 5))
plt.plot(week_df["Timestamp"], week_df["Shared_Perfect"],
         label="Theoretical optimum", color="green", linestyle="--", linewidth=2)
plt.plot(week_df["Timestamp"], week_df["Shared_MPC"],
         label="Closed-Loop (AMPC)", color="blue", linewidth=2, alpha=0.8)
plt.plot(week_df["Timestamp"], week_df["Shared_Open"],
         label="Open-Loop", color="red", linestyle="-.", linewidth=1)
plt.title("Shared energy in the critical week of August (recovery through AMPC)")
plt.ylabel("Shared energy [kWh]")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
save_figure(fig2, "shared_energy_critical_week_MPC")
plt.close(fig2)

print(f"\n>>> Total elapsed time: {time.time() - start_time:.1f} s")
