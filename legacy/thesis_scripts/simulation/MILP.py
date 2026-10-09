# MILP example: optimal dispatch of a battery coupled with PV under time-varying feed-in prices
# Extended version: supports 8760-hour (1-year) PV profile or daily pattern repeated over a year

import pulp
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
import os

# -------------------------
# Parameters and inputs
# -------------------------
T_day = 24  # hours per day
N_days = 365  # one year
T = T_day * N_days  # total hours

# --- Example daily PV profile (kW) over 24 hours ---
pv_day = np.array([0,0,0,0,0,0.1,0.5,1.2,2.5,3.5,4.0,3.8,3.2,2.0,1.0,0.6,0.2,0,0,0,0,0,0,0])

# --- Repeat daily PV profile for the entire year ---
#pv = np.tile(pv_day, N_days)

def get_pvgis_pv(lat=37.257, lon=15.171, year=2023, capacity_kwp=1):
    url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "pvtechchoice": "crystSi",
        "peakpower": capacity_kwp,
        "loss": 10,
        "angle": 33,
        "aspect": -3,
        "pvcalculation": 1,
        "outputformat": "csv",
        "timeframe": "hourly"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"PVGIS API request failed with code {response.status_code}")

    # Save locally
    csv_filename = "pvgis_data.csv"
    with open(csv_filename, "w", newline="") as csv_file:
        csv_file.write(response.text)

    # Read only the relevant data (8760 hours)
    df = pd.read_csv(csv_filename, skiprows=10, nrows=8760)
    power_W = df.iloc[:, 1].values  # second column is PV power [W]
    power_kW = power_W / 1000.0     # convert to kW

    return power_kW

# --- Retrieve PV production data (8760 hourly values) ---
# === User inputs (optional, default values provided) ===
try:
    lat = float(input("Enter latitude [default 37.257]: ") or 37.257)
    lon = float(input("Enter longitude [default 15.171]: ") or 15.171)
    year = int(input("Enter year [default 2023]: ") or 2023)
    capacity_kwp = float(input("Enter PV system capacity in kWp [default 1.0]: ") or 1.0)
except ValueError:
    print("Invalid input detected. Using default parameters.")
    lat, lon, year, capacity_kwp = 37.257, 15.171, 2023, 1.0

# --- Retrieve PV production data (8760 hourly values) ---
pv = get_pvgis_pv(lat=lat, lon=lon, year=year, capacity_kwp=capacity_kwp)
T = len(pv)


# --- Example feed-in price (EUR/kWh), same each day for now ---
price_day = np.array([
    0.18,0.20,0.22,0.18,0.10,0.07,0.06,0.06,0.07,0.08,0.09,0.10,
    0.12,0.14,0.16,0.20,0.24,0.26,0.22,0.18,0.10,0.08,0.06,0.06
])
#price = np.tile(price_day, N_days)

# --- Path to Excel file on Desktop ---
# Works for Windows, macOS, Linux
#desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
desktop_path = "local_data"
excel_file = os.path.join(desktop_path, "PrezziZonali_SUD.xlsx")

print(f"📂 Loading price data from: {excel_file}")



try:
    # Read the first column (no header)
    df_price = pd.read_excel(excel_file, header=None)
    price = df_price.iloc[:, 0].values  # get first column as numpy array
    price = price / 1000.0

    # Check expected length (8760 hours)
    if len(price) != 8760:
        raise ValueError(f"Expected 8760 hourly prices, found {len(price)}")

    print(f"✅ Loaded {len(price)} hourly price values from Excel file.")

except FileNotFoundError:
    print(f"❌ File not found at {excel_file}")
    print("Please make sure 'prices.xlsx' is on your Desktop.")
    price = np.zeros(8760)  # fallback placeholder

except Exception as e:
    print(f"⚠️ Error reading Excel file: {e}")
    price = np.zeros(8760)  # fallback placeholder

# Battery specs
P_max = 1.0      # max charge/discharge power (kW)
E_max = 5.0      # battery energy capacity (kWh)
soc_init = 1.0   # initial state of charge (kWh)
soc_min = 0.1*E_max
soc_max = E_max
eta_c = 0.95     # charging efficiency
eta_d = 0.95     # discharging efficiency

# -------------------------
# Build MILP with PuLP
# -------------------------
model = pulp.LpProblem('Battery_PV_dispatch_year', pulp.LpMaximize)

# Decision variables
charge = pulp.LpVariable.dicts('charge', range(T), lowBound=0, upBound=P_max, cat='Continuous')
discharge = pulp.LpVariable.dicts('discharge', range(T), lowBound=0, upBound=P_max, cat='Continuous')
export = pulp.LpVariable.dicts('export', range(T), lowBound=0, cat='Continuous')
soc = pulp.LpVariable.dicts('soc', range(T), lowBound=soc_min, upBound=soc_max, cat='Continuous')
# Binary variable to prevent simultaneous charge & discharge
is_discharging = pulp.LpVariable.dicts('is_discharging', range(T), cat='Binary')

# Objective: maximize total revenue from exported energy
model += pulp.lpSum([price[t] * export[t] for t in range(T)])

# Constraints
for t in range(T):
    model += export[t] == pv[t] - charge[t] + discharge[t]
    model += charge[t] <= pv[t]
    model += charge[t] <= P_max * (1 - is_discharging[t])
    model += discharge[t] <= P_max * is_discharging[t]

# SOC dynamics
for t in range(T):
    if t == 0:
        model += soc[t] == soc_init + eta_c * charge[t] - discharge[t] / eta_d
    else:
        model += soc[t] == soc[t-1] + eta_c * charge[t] - discharge[t] / eta_d

# End-of-year SOC target
model += soc[T-1] >= max(soc_min, soc_init - 0.01)

# Solve MILP
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

print('Status:', pulp.LpStatus[model.status])

# Collect results
results = pd.DataFrame({
    'pv_kW': pv,
    'price_EUR_per_kWh': price,
    'charge_kW': [pulp.value(charge[t]) for t in range(T)],
    'discharge_kW': [pulp.value(discharge[t]) for t in range(T)],
    'export_kW': [pulp.value(export[t]) for t in range(T)],
    'soc_kWh': [pulp.value(soc[t]) for t in range(T)],
})

results['revenue_EUR'] = results['export_kW'] * results['price_EUR_per_kWh']

# --- Summary ---
print(f"\nTotal annual revenue EUR: {results['revenue_EUR'].sum():.2f}")
print('Average daily revenue EUR:', results['revenue_EUR'].sum() / N_days)

# --- Optional plots for one day sample (first day) ---
day_range = range(0, 24)
plt.figure(figsize=(9,4))
plt.plot(day_range, results['pv_kW'].iloc[day_range], label='PV [kW]', linewidth=2)
plt.plot(day_range, results['charge_kW'].iloc[day_range], label='Charge [kW]', linewidth=2)
plt.plot(day_range, results['discharge_kW'].iloc[day_range], label='Discharge [kW]', linewidth=2)
plt.plot(day_range, results['soc_kWh'].iloc[day_range], label='SOC [kWh]', linewidth=2)
plt.xlabel('Hour of Day')
plt.ylabel('Power / Energy')
plt.title('Battery and PV Operation - Sample Day')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Save results to CSV
results.to_csv('dispatch_results_year.csv', index_label='hour')
print('\nResults saved to dispatch_results_year.csv')

# --- Integration note ---
# The variable `pv` can be replaced by data from your PVGIS API script once available.
# Example:
# pv = your_pvgis_function_output  # must be a NumPy array of length 8760

# Variables to plot
variables = ['pv_kW', 'price_EUR_per_kWh', 'charge_kW', 'discharge_kW', 'export_kW']
titles = ['PV Production [kW]', 'Price [€/kWh]', 'Battery Charge [kW]',
          'Battery Discharge [kW]', 'Exported Power [kW]']

# Create figure: 3 rows x 2 columns
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()

for i, var in enumerate(variables):
    data = results[var].values.reshape((365, 24))
    im = axes[i].imshow(data, aspect='auto', origin='upper', cmap='plasma',
                        extent=[0, 24, 0, 365])
    axes[i].set_title(titles[i], fontsize=10)
    axes[i].set_ylabel('Day of Year', fontsize=8)
    axes[i].set_xlabel('Hour of Day', fontsize=8)
    axes[i].tick_params(axis='both', labelsize=8)
    fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)

# 6th subplot empty
axes[5].axis('off')
axes[5].set_title('Unused / Notes', fontsize=10)

plt.tight_layout()
plt.show()
