# MILP example: optimal dispatch of a battery coupled with PV under time-varying feed-in prices
# Extended version: supports 8760-hour (1-year) PV profile or daily pattern repeated over a year

import pulp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ----------------------------------------------------------------
# 1. PARAMETRI E INPUT
# ----------------------------------------------------------------
T_day = 24  # hours al day
N_days = 365  # un anno
N_hours_year = T_day * N_days # Ore totali in un anno standard

# --- Parameters BESS (basati sullo script MATLAB) ---
try:
    n_BESS = int(input("Inserisci il numero di BESS da installare: ") or 1)
except ValueError:
    print("Input non valido. Utilizzo il valore di default: 1 BESS.")
    n_BESS = 1

# Parameters per un singolo BESS
E_nom_singolo = 760.0       # Capacità nominale [kWh]
P_charge_singolo = 380.0    # Potenza di charge [kW]
P_discharge_singolo = 380.0 # Potenza di discharge [kW]

# Parameters totali calcolati in base al numero di BESS
E_nom_total = E_nom_singolo * n_BESS
P_bess_max = P_charge_singolo * n_BESS

# Efficienza e limiti di Stato di Carica (SoC)
eta_BESS = 0.95
SoC_min_perc = 0.10
SoC_max_perc = 1.00
SoC_iniziale_perc = 0.50

# Computation dei valori assoluti per il modello
soc_min_kWh = E_nom_total * SoC_min_perc
soc_max_kWh = E_nom_total * SoC_max_perc
soc_init_kWh = E_nom_total * SoC_iniziale_perc
eta_cycle = np.sqrt(eta_BESS)

# Stampa di riepilogo dello scenario
scenario_txt = f'Scenario ARBITRAGGIO con {n_BESS}x BESS ({E_nom_total:.0f}kWh/{P_bess_max:.0f}kW)'
print("\n" + "="*80)
print(f">>> {scenario_txt.upper()}")
print(f">>> Parameters BESS totali: E={E_nom_total:.0f}kWh, P_charge/discharge={P_bess_max:.0f}kW")
print("="*80 + "\n")


# --- Loading data di production PV dal file CSV di PVsyst ---
file_produzione = "local_data/CaseStudy_Project_VC0_HourlyRes_0.CSV"
print(f"📂 Loading data di production PV da: {file_produzione}")

try:
    df_pv = pd.read_csv(
        file_produzione,
        delimiter=';',
        skiprows=11,
        header=None,
        names=['Timestamp_str', 'E_Grid'],
        decimal=','
    )
    # --- MODIFICA CHIAVE: Assicura che la colonna sia di tipo numerico ---
    # `errors='coerce'` trasformerà qualsiasi valore non numerico in 'NaN' (Not a Number)
    df_pv['E_Grid'] = pd.to_numeric(df_pv['E_Grid'], errors='coerce')
    # Sostituisci eventuali NaN con 0, assumendo che un dato mancante significhi production nulla
    df_pv.fillna(0, inplace=True)

    pv = df_pv['E_Grid'].values
    print(f"✅ Caricati e convertiti in formato numerico {len(pv)} valori di production PV.")

except FileNotFoundError:
    print(f"❌ ERRORE: File not found al percorso {file_produzione}")
    exit()
except Exception as e:
    print(f"⚠️ ERRORE durante la lettura del file CSV: {e}")
    exit()


# --- Loading prices zonali da file Excel ---
desktop_path = "local_data"
excel_file = os.path.join(desktop_path, "PrezziZonali_SUD.xlsx")

print(f"📂 Loading data dei prices da: {excel_file}")

try:
    df_price = pd.read_excel(excel_file, header=None)
    price = df_price.iloc[:, 0].values
    price = price / 1000.0  # Converti da EUR/MWh a EUR/kWh
    print(f"✅ Caricati {len(price)} valori orari dei prices dal file Excel.")
except FileNotFoundError:
    print(f"❌ ERRORE: File not found al percorso {excel_file}")
    exit()
except Exception as e:
    print(f"⚠️ ERRORE durante la lettura del file Excel: {e}")
    exit()

# --- Sincronizzazione della lunghezza dei data ---
print("\n🔄 Sincronizzazione della lunghezza dei data di input...")
if len(pv) > N_hours_year:
    print(f"⚠️ Dati PV ({len(pv)}) più lunghi di {N_hours_year}. Verranno troncati.")
    pv = pv[:N_hours_year]
if len(price) > N_hours_year:
    print(f"⚠️ Dati Prezzi ({len(price)}) più lunghi di {N_hours_year}. Verranno troncati.")
    price = price[:N_hours_year]

if len(pv) != N_hours_year or len(price) != N_hours_year:
    print(f"❌ ERRORE: La lunghezza dei data non è 8760 dopo il loading.")
    exit()

T = N_hours_year
print(f"✅ Dati sincronizzati. La simulazione verrà eseguita per {T} hours.\n")
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 2. COSTRUZIONE DEL MODELLO MILP CON PuLP
# ----------------------------------------------------------------
model = pulp.LpProblem('Battery_PV_dispatch_year', pulp.LpMaximize)

# Variabili di decisione
charge = pulp.LpVariable.dicts('charge', range(T), lowBound=0, upBound=P_bess_max, cat='Continuous')
discharge = pulp.LpVariable.dicts('discharge', range(T), lowBound=0, upBound=P_bess_max, cat='Continuous')
export = pulp.LpVariable.dicts('export', range(T), lowBound=0, cat='Continuous')
soc = pulp.LpVariable.dicts('soc', range(T), lowBound=soc_min_kWh, upBound=soc_max_kWh, cat='Continuous')
is_discharging = pulp.LpVariable.dicts('is_discharging', range(T), cat='Binary')

# Funzione obiettivo
model += pulp.lpSum([price[t] * export[t] for t in range(T)])

# Vincoli
for t in range(T):
    # --- MODIFICA CHIAVE: Riorganizzazione del vincolo per stabilità ---
    # Bilancio energetico: Potenza Esportata + Potenza in Carica - Potenza in Scarica = Potenza PV
    model += export[t] + charge[t] - discharge[t] == pv[t]

    model += charge[t] <= pv[t]
    model += charge[t] <= P_bess_max * (1 - is_discharging[t])
    model += discharge[t] <= P_bess_max * is_discharging[t]

# Dinamica dello Stato di Carica (SoC)
for t in range(T):
    if t == 0:
        model += soc[t] == soc_init_kWh + eta_cycle * charge[t] - discharge[t] / eta_cycle
    else:
        model += soc[t] == soc[t-1] + eta_cycle * charge[t] - discharge[t] / eta_cycle

# Vincolo sul SoC alla end dell'anno
model += soc[T-1] >= max(soc_min_kWh, soc_init_kWh - 0.01 * E_nom_total)

# ----------------------------------------------------------------
# 3. RISOLUZIONE E ANALISI DEI RISULTATI
# ----------------------------------------------------------------
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

print('Stato della soluzione:', pulp.LpStatus[model.status])

if pulp.LpStatus[model.status] == 'Optimal':
    results = pd.DataFrame({
        'pv_kW': pv,
        'price_EUR_per_kWh': price,
        'charge_kW': [pulp.value(charge[t]) for t in range(T)],
        'discharge_kW': [pulp.value(discharge[t]) for t in range(T)],
        'export_kW': [pulp.value(export[t]) for t in range(T)],
        'soc_kWh': [pulp.value(soc[t]) for t in range(T)],
    })

    results['revenue_EUR'] = results['export_kW'] * results['price_EUR_per_kWh']

    total_revenue = results['revenue_EUR'].sum()
    print(f"\nRicavo annual totale: {total_revenue:.2f} EUR")
    print(f"Ricavo medio giornaliero: {total_revenue / N_days:.2f} EUR")

    results.to_csv('dispatch_results_year.csv', index_label='hour')
    print('\nResults salvati nel file dispatch_results_year.csv')

    # --- Figures (invariati) ---
    day_range = range(0, 24)
    plt.figure(figsize=(12, 6))
    plt.plot(day_range, results['pv_kW'].iloc[day_range], label='Produzione PV [kW]', linewidth=2)
    plt.plot(day_range, results['charge_kW'].iloc[day_range], 'g--', label='Carica BESS [kW]', linewidth=2)
    plt.plot(day_range, results['discharge_kW'].iloc[day_range], 'r--', label='Scarica BESS [kW]', linewidth=2)
    plt.plot(day_range, results['soc_kWh'].iloc[day_range], label='SoC [kWh]', linewidth=2, color='purple')
    plt.xlabel('Ora del Giorno')
    plt.ylabel('Potenza [kW] / Energia [kWh]')
    plt.title('Funzionamento PV e BESS - Giorno Campione')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    variables = ['pv_kW', 'price_EUR_per_kWh', 'charge_kW', 'discharge_kW', 'export_kW', 'soc_kWh']
    titles = ['Produzione PV [kW]', 'Prezzo [€/kWh]', 'Carica Batteria [kW]',
              'Scarica Batteria [kW]', 'Potenza Esportata [kW]', 'Stato di Carica [kWh]']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        data = results[var].values.reshape((365, 24))
        im = axes[i].imshow(data, aspect='auto', origin='upper', cmap='viridis', extent=[0, 24, 365, 0])
        axes[i].set_title(titles[i], fontsize=10)
        axes[i].set_ylabel('Giorno dell\'Anno', fontsize=8)
        axes[i].set_xlabel('Ora del Giorno', fontsize=8)
        axes[i].tick_params(axis='both', labelsize=8)
        fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.suptitle('Riepilogo Annuale del Funzionamento', fontsize=16, y=1.02)
    plt.show()

else:
    print("Il modello non ha trovato una soluzione ottimale.")