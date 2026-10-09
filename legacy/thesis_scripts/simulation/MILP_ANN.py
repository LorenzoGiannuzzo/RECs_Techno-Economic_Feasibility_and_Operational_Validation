# -*- coding: utf-8 -*-
"""
Script di Ottimizzazione MILP (Fase 2: Forecast-based Operation)
Ottimizza la strategia della battery basandosi sul carico RESIDENZIALE PREDETTO
dalla Rete Neurale (ANN) + i carichi REALI degli altri settori (Smart Meter).
"""
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus
import matplotlib.pyplot as plt

print(">>> Start Ottimizzazione MILP (Base Previsionale)...")

# =============================================================================
# 1. IMPOSTAZIONI E CARICAMENTO DATI
# =============================================================================
n_BESS = 1
pv_plant_power = 1000

base_incentivo = 60
cap_incentivo = 100
correzione_geografica = 0
arera_fee = 10.57
usa_incentivo_PNRR = False
fattore_riduzione_PNRR = 0.6

# --- NOVITÀ: Loading e Aggregazione Dati Ibrida ---
print(">>> Loading profili di carico (Reali + Predetto ANN)...")
try:
    df_pa = pd.read_excel("CER_Carico_PA.xlsx")
    df_illuminazione = pd.read_excel("CER_Carico_Illuminazione.xlsx")
    df_industria = pd.read_excel("CER_Carico_Industria.xlsx")

    # Questo è il file che hai appena generato con la Rete Neurale!
    df_residenziale_pred = pd.read_excel("CER_Carico_Residenziale_PREDETTO_25.xlsx")

    # Creiamo il DataFrame delle categories per eventuali figures
    consumi_categorie_df = pd.DataFrame({
        'Pubblica Amm.': df_pa['Carico_kWh'].values,
        'Illuminazione': df_illuminazione['Carico_kWh'].values,
        'Industria': df_industria['Carico_kWh'].values,
        'Residenziale (Predetto)': df_residenziale_pred['Carico_kWh'].values
    })

    # Somma totale oraria (il nuovo rec_consumption su cui lavorerà il MILP)
    rec_consumption = consumi_categorie_df.sum(axis=1).values
    print(">>> Aggregazione ibrida completata successfully.")

except FileNotFoundError as e:
    print(f"ERRORE: File not found - {e}.")
    exit()

# --- Loading Prezzi e Produzione (Invariato) ---
try:
    df_pz = pd.read_excel("MGP_PrezziZonali_SUD.xlsx", decimal=',', header=0)
    df_pz.columns = ['Data', 'Ora', 'Prezzo_MWh']
    df_pz['Timestamp'] = pd.to_datetime(df_pz['Data'], dayfirst=True) + pd.to_timedelta(df_pz['Ora'] - 1, unit='h')
    df_pz = df_pz.set_index('Timestamp').sort_index()
    df_pz_2025 = df_pz.loc['2025-01-01':'2025-12-31 23:59:59']
    Pz_orario = df_pz_2025['Prezzo_MWh'].values

    file_produzione = "local_data/CaseStudy_Project_VC0_HourlyRes_0.CSV"
    df_produzione = pd.read_csv(file_produzione, sep=';', skiprows=10, header=None, usecols=[1],
                                names=['Produzione_str'], dtype=str)
    pv_production = pd.to_numeric(df_produzione['Produzione_str'].str.replace(',', '.'), errors='coerce').fillna(
        0).values[:8760]
except FileNotFoundError as e:
    print(f"ERRORE: File not found - {e}");
    exit()

# --- Parameters BESS ---
E_nom_singolo = 760;
P_charge_singolo = 380;
P_discharge_singolo = 380
E_nom = E_nom_singolo * n_BESS;
P_charge_max = P_charge_singolo * n_BESS;
P_discharge_max = P_discharge_singolo * n_BESS
eta_BESS = 0.95;
SoC_min = 0.10;
SoC_max = 1.00;
SoC_initial = 0.5
E_min = E_nom * SoC_min;
E_max = E_nom * SoC_max;
eta_cycle = eta_BESS ** 0.5
n_hours = 8760;
hours = range(n_hours)

# =============================================================================
# 2 & 3. DEFINIZIONE E RISOLUZIONE DEL MILP
# =============================================================================
print(">>> Costruzione modello MILP...")
prob = LpProblem("Max_Ricavi_CER_Previsionale", LpMaximize)

E_batt = LpVariable.dicts("E", hours, lowBound=E_min, upBound=E_max)
P_carica_da_FV = LpVariable.dicts("Pc", hours, lowBound=0, upBound=P_charge_max)
P_scarica = LpVariable.dicts("Pd", hours, lowBound=0, upBound=P_discharge_max)
P_immissione_da_FV = LpVariable.dicts("Pifv", hours, lowBound=0)
P_immissione_da_BESS = LpVariable.dicts("Pibess", hours, lowBound=0)
P_prelievo = LpVariable.dicts("Pp", hours, lowBound=0)
E_shared = LpVariable.dicts("Ec", hours, lowBound=0)
is_charging = LpVariable.dicts("d", hours, cat=LpBinary)

hourly_incentive_tariff = [min(cap_incentivo, base_incentivo + max(0, 180 - p) + correzione_geografica) * (
    fattore_riduzione_PNRR if usa_incentivo_PNRR else 1) for p in Pz_orario]

prob += lpSum((E_shared[h] / 1000) * (hourly_incentive_tariff[h] + arera_fee) + (
            (P_immissione_da_FV[h] + P_immissione_da_BESS[h]) / 1000) * Pz_orario[h] for h in hours), "Ricavo_Predetto"

E_batt_iniziale = E_nom * SoC_initial
for h in hours:
    prob += pv_production[h] == P_carica_da_FV[h] + P_immissione_da_FV[h]
    prob += P_scarica[h] == P_immissione_da_BESS[h]
    prob += rec_consumption[h] == E_shared[h] + P_prelievo[h]
    if h == 0:
        prob += E_batt[h] == E_batt_iniziale + P_carica_da_FV[h] * eta_cycle - P_scarica[h] / eta_cycle
    else:
        prob += E_batt[h] == E_batt[h - 1] + P_carica_da_FV[h] * eta_cycle - P_scarica[h] / eta_cycle
    prob += P_carica_da_FV[h] <= P_charge_max * is_charging[h]
    prob += P_scarica[h] <= P_discharge_max * (1 - is_charging[h])
    prob += E_shared[h] <= rec_consumption[h]
    prob += E_shared[h] <= P_immissione_da_FV[h] + P_immissione_da_BESS[h]

print(">>> Risoluzione in progress...")
prob.solve()
print(f">>> Stato: {LpStatus[prob.status]}")

# =============================================================================
# 4. SALVATAGGIO DELLA STRATEGIA OPERATIVA (IL PIANO)
# =============================================================================
if prob.status == 1:
    # Salviamo le "regole" che ha deciso il MILP (il piano di charge/discharge)
    risultati_previsionali = pd.DataFrame({
        'Consumo_CER_Predetto_kWh': rec_consumption,
        'Carica_Batteria_Pianificata_kWh': [P_carica_da_FV[h].varValue for h in hours],
        'Scarica_Batteria_Pianificata_kWh': [P_scarica[h].varValue for h in hours],
        'SoC_Batteria_Pianificato_Percentuale': [E_batt[h].varValue / E_nom * 100 for h in hours],
    })

    # Salviamo in un file Excel che sarà l'input per lo Step 3 (La simulazione real)
    nome_file_piano = f"Piano_Operativo_BESS_{n_BESS}x_PREVISIONALE.xlsx"
    risultati_previsionali.to_excel(nome_file_piano, index=False)

    # Stampiamo solo il revenue *teorico* atteso
    ricavo_teorico = sum((E_shared[h].varValue / 1000) * (hourly_incentive_tariff[h] + arera_fee) + (
                (P_immissione_da_FV[h].varValue + P_immissione_da_BESS[h].varValue) / 1000) * Pz_orario[h] for h in hours)
    print(f"\n--- RICAVO TEORICO ATTESO SULLE PREVISIONI: € {ricavo_teorico:,.2f} ---")
    print(f">>> Piano operativo salvato in {nome_file_piano}. Pronto per lo Step 3!")
else:
    print("Fallimento ottimizzazione.")