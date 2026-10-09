# -*- coding: utf-8 -*-
"""
Real Operation Simulation (Step 3)
Revenue computation compliant with the Italian regulation (RID on the whole injection).
"""
import os
import pandas as pd
import numpy as np
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


print(">>> Starting Operational Simulation (Italian regulation)...")

# =============================================================================
# 1. DATA LOADING
# =============================================================================
try:
    optimum_df = pd.read_excel("Risultati_Ottimizzazione_StandAlone_1xBESS.xlsx",
                               sheet_name="Dati Orari").iloc[:8760]
    real_consumption = optimum_df["Consumo_CER_kWh"].values
    pv_production = optimum_df["Produzione_FV_kWh"].values
    zonal_price = optimum_df["Pz_Eur_MWh"].values
    hourly_incentive_tariff = optimum_df["Tariffa_Incentivo_Oraria_Eur_MWh"].values

    plan_df = pd.read_excel("Piano_Operativo_BESS_1x_PREVISIONALE.xlsx")
    planned_charge = plan_df["Carica_Batteria_Pianificata_kWh"].values[:8760]
    planned_discharge = plan_df["Scarica_Batteria_Pianificata_kWh"].values[:8760]

    if "Timestamp" in optimum_df.columns:
        timestamp_index = pd.to_datetime(optimum_df["Timestamp"])
    else:
        timestamp_index = pd.date_range(start="2025-01-01 00:00", periods=8760, freq="h")
except Exception as e:
    print(f"LOADING ERROR: {e}")
    exit()

# Technical parameters
arera_fee = 10.57
E_nom = 760
P_max = 380
eta_BESS = 0.95
eta_cycle = eta_BESS ** 0.5
SoC_min = 0.10
SoC_max = 1.00
SoC_initial = 0.5
E_min = E_nom * SoC_min
E_max = E_nom * SoC_max

# =============================================================================
# 2. PHYSICAL SIMULATION
# =============================================================================
n_hours = 8760
real_shared_energy = np.zeros(n_hours)
real_total_injection = np.zeros(n_hours)
real_SoC = np.zeros(n_hours + 1)
real_SoC[0] = SoC_initial * E_nom

for h in range(n_hours):
    max_discharge = (real_SoC[h] - E_min) * eta_cycle
    max_charge = (E_max - real_SoC[h]) / eta_cycle

    eff_discharge = min(planned_discharge[h], max_discharge)
    eff_charge = min(planned_charge[h], max_charge)

    # Energy physically injected into the grid (PV - charge + discharge)
    grid_injection = max(0, (pv_production[h] - eff_charge) + eff_discharge)

    # Virtually shared energy (minimum between injection and withdrawal/consumption)
    real_shared_energy[h] = min(real_consumption[h], grid_injection)
    real_total_injection[h] = grid_injection

    real_SoC[h + 1] = real_SoC[h] + eff_charge * eta_cycle - eff_discharge / eta_cycle

# =============================================================================
# 3. CORRECT ECONOMIC COMPUTATION (REC regulation)
# =============================================================================
# Perfect Foresight (values taken from the Excel)
perfect_revenue = (optimum_df["Ricavo_Incentivo_Orario_Eur"] +
                   optimum_df["Ricavo_ARERA_Orario_Eur"] +
                   optimum_df["Ricavo_RID_Orario_Eur"]).sum()

# Real (compliant: incentive on shared energy + RID on the WHOLE injection)
real_incentive_revenue = (real_shared_energy / 1000) * (hourly_incentive_tariff + arera_fee)
real_rid_revenue = (real_total_injection / 1000) * zonal_price
real_total_revenue = np.sum(real_incentive_revenue) + np.sum(real_rid_revenue)

print("\n" + "=" * 55)
print(" FINAL RESULTS OF THE ROBUSTNESS TEST")
print("=" * 55)
print(f"1. Theoretical optimum (Perfect Foresight): EUR {perfect_revenue:,.2f}")
print(f"2. Real revenue (Open-Loop with ANN):       EUR {real_total_revenue:,.2f}")
print("-" * 55)
revenue_loss = perfect_revenue - real_total_revenue
print(f"REAL ECONOMIC LOSS: EUR {revenue_loss:,.2f} ({(revenue_loss / perfect_revenue) * 100:.2f} %)")
print("=" * 55)

# =============================================================================
# 4. FIGURES
# =============================================================================
plot_df = pd.DataFrame({
    "Hours": range(n_hours),
    "Timestamp": timestamp_index,
    "Perf_Cum": (optimum_df["Ricavo_Incentivo_Orario_Eur"] +
                 optimum_df["Ricavo_ARERA_Orario_Eur"] +
                 optimum_df["Ricavo_RID_Orario_Eur"]).cumsum(),
    "Real_Cum": pd.Series(real_incentive_revenue + real_rid_revenue).cumsum(),
})

fig = plt.figure(figsize=(10, 6))
plt.plot(plot_df["Hours"], plot_df["Perf_Cum"], label="Theoretical optimum", color="green")
plt.plot(plot_df["Hours"], plot_df["Real_Cum"], label="Real (Open-Loop)", color="red", linestyle="--")
plt.fill_between(plot_df["Hours"], plot_df["Perf_Cum"], plot_df["Real_Cum"], color="red", alpha=0.1)
plt.title("Cumulative REC revenue: robustness analysis")
plt.xlabel("Hour of the year")
plt.ylabel("Euro [EUR]")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_figure(fig, "cumulative_REC_revenue")
plt.close(fig)
