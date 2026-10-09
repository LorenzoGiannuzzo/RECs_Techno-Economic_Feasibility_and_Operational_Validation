import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# -------------------------------------------------------------------------
# SCRIPT PER L'ANALISI FINANZIARIA DELLA CER (v3.0 - Sensitivity BESS)
# -------------------------------------------------------------------------
# Calcola PBP e VAN considerando come unico revenue la vendita di energy (RID),
# introducendo un'analisi di sensibilità sul CAPEX del BESS.
# -------------------------------------------------------------------------

# %% 1. IMPOSTAZIONI E DEFINIZIONE DEGLI SCENARI
print('>>> Start Analisi Finanziaria con Sensitivity (basata su RID)...\n')

cartella_file = "local_data"

scenarios = [
    {'Nome': 'Senza BESS (Autoconsumo)', 'FileMAT': 'Dati_AUTOCONSUMO_0xBESS.mat', 'NumBESS': 0, 'Colore': 'b',
     'Stile': '-o'},
    {'Nome': '1 BESS (Autoconsumo)', 'FileMAT': 'Dati_AUTOCONSUMO_1xBESS.mat', 'NumBESS': 1, 'Colore': 'g',
     'Stile': '-o'},
    {'Nome': '1 BESS (Arbitraggio)', 'FileMAT': 'Dati_ARBITRAGGIO_1xBESS.mat', 'NumBESS': 1, 'Colore': 'm',
     'Stile': '-o'},
    {'Nome': '1 BESS (Ottimizzazione)', 'FileMAT': 'Dati_OTTIMIZZAZIONE_1xBESS.mat', 'NumBESS': 1, 'Colore': 'r',
     'Stile': '--s'},
]

# Parameters finanziari e Vettori per la Sensitivity
pv_plant_power_kW = 1000
costo_unitario_FV = 800
costi_unitari_BESS = [250, 350, 450, 550]  # Valori per la sensitivity
capacita_BESS_singolo = 760
OPEX_percentuale = 1.5
contributo_PNRR_percentuale = 0
durata_analisi = 20
tasso_attualizzazione = 4

final_results = []

# %% 2. CICLO DI CALCOLO PER OGNI SCENARIO E PER OGNI COSTO BESS
for i, current_scenario in enumerate(scenarios):
    print(f"--- Elaborazione Scenario {i + 1}: {current_scenario['Nome']} ---")
    try:
        percorso_completo = os.path.join(cartella_file, current_scenario['FileMAT'])
        loaded_data = loadmat(percorso_completo)
        # Ricavo annual basato solo sul Ritiro Dedicato (RID)
        annual_revenue = loaded_data['annual_rid_revenue'][0][0]
    except FileNotFoundError:
        print(f"ATTENZIONE: Impossibile trovare il file {current_scenario['FileMAT']}. Salto lo scenario.")
        continue
    except KeyError:
        print(
            f"ATTENZIONE: Variabile 'annual_rid_revenue' non trovata in {current_scenario['FileMAT']}. Salto lo scenario.")
        continue

    # Eseguiamo l'analisi di sensibilità sui prices del BESS
    for costo_BESS in costi_unitari_BESS:
        # Se non c'è il BESS, il suo costo è irrilevante: calcoliamo i data una sola volta per non duplicarli
        if current_scenario['NumBESS'] == 0 and costo_BESS != costi_unitari_BESS[0]:
            continue

        # Computation CAPEX
        CAPEX_FV = pv_plant_power_kW * costo_unitario_FV
        CAPEX_BESS = current_scenario['NumBESS'] * capacita_BESS_singolo * costo_BESS
        CAPEX_totale = CAPEX_FV + CAPEX_BESS
        contributo_PNRR = CAPEX_totale * (contributo_PNRR_percentuale / 100)
        CAPEX_netto = CAPEX_totale - contributo_PNRR

        # Computation OPEX
        OPEX_annuale = CAPEX_totale * (OPEX_percentuale / 100)

        # Computation flussi di cash
        discounted_cash_flow = np.zeros(durata_analisi + 1)
        discounted_cash_flow[0] = -CAPEX_netto
        annual_net_flow = annual_revenue - OPEX_annuale

        for t in range(1, durata_analisi + 1):
            discounted_cash_flow[t] = annual_net_flow / (1 + tasso_attualizzazione / 100) ** t

        cumulative_discounted_flow = np.cumsum(discounted_cash_flow)
        VAN = np.sum(discounted_cash_flow)

        # Computation Payback Period (PBP)
        idx_pbp_array = np.where(cumulative_discounted_flow > 0)[0]
        if len(idx_pbp_array) > 0:
            idx_pbp = idx_pbp_array[0]
            PBP = (idx_pbp - 1) + abs(cumulative_discounted_flow[idx_pbp - 1]) / discounted_cash_flow[idx_pbp]
        else:
            PBP = np.inf

        final_results.append({
            'Nome': current_scenario['Nome'],
            'Costo_Unitario_BESS': costo_BESS if current_scenario['NumBESS'] > 0 else 'N/A',
            'VAN': VAN,
            'PBP': PBP,
            'Colore': current_scenario['Colore'],
            'Stile': current_scenario['Stile'],
            'NumBESS': current_scenario['NumBESS']
        })
    print('Calcoli per sensitivity completati.\n')

# %% 3. CREAZIONE GRAFICI DI SENSITIVITY (VAN e PBP)
print('>>> Generazione dei figures di Sensitivity...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Filtro gli scenarios con e senza BESS
scenari_con_bess = [s for s in scenarios if s['NumBESS'] > 0]
scenari_senza_bess = [s for s in scenarios if s['NumBESS'] == 0]

# Traccio i figures per chi ha il BESS
for scenario in scenari_con_bess:
    scenario_data = [r for r in final_results if r['Nome'] == scenario['Nome']]
    if not scenario_data: continue

    costi_x = [r['Costo_Unitario_BESS'] for r in scenario_data]
    van_y = [r['VAN'] for r in scenario_data]
    pbp_y = [r['PBP'] for r in scenario_data]

    ax1.plot(costi_x, van_y, scenario['Stile'], linewidth=2, color=scenario['Colore'],
             markersize=8, label=scenario['Nome'])
    ax2.plot(costi_x, pbp_y, scenario['Stile'], linewidth=2, color=scenario['Colore'],
             markersize=8, label=scenario['Nome'])

# Aggiungo linee costanti per lo scenario Senza BESS (come baseline di confronto)
for scenario in scenari_senza_bess:
    scenario_data = [r for r in final_results if r['Nome'] == scenario['Nome']]
    if not scenario_data: continue

    van_base = scenario_data[0]['VAN']
    pbp_base = scenario_data[0]['PBP']

    ax1.axhline(van_base, color=scenario['Colore'], linestyle='--', linewidth=2, label=scenario['Nome'] + ' (Baseline)')
    if np.isfinite(pbp_base):
        ax2.axhline(pbp_base, color=scenario['Colore'], linestyle='--', linewidth=2,
                    label=scenario['Nome'] + ' (Baseline)')

# Formattazione asse VAN
ax1.set_title('Sensitivity: Valore Attuale Netto (VAN) vs Costo BESS', fontsize=14, fontweight='bold')
ax1.set_xlabel('Costo Unitario BESS (€/kWh)', fontsize=12)
ax1.set_ylabel('VAN (€)', fontsize=12)
ax1.set_xticks(costi_unitari_BESS)
ax1.grid(True)
ax1.legend()

# Formattazione asse PBP
ax2.set_title('Sensitivity: Payback Period (PBP) vs Costo BESS', fontsize=14, fontweight='bold')
ax2.set_xlabel('Costo Unitario BESS (€/kWh)', fontsize=12)
ax2.set_ylabel('Payback Period (Anni)', fontsize=12)
ax2.set_xticks(costi_unitari_BESS)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# %% 4. STAMPA TABELLA RIASSUNTIVA FINALE
print('\n\n--- TABELLA RIASSUNTIVA SENSITIVITY COSTO BESS ---')
tabella_risultati = pd.DataFrame({
    'Scenario': [r['Nome'] for r in final_results],
    'Costo_BESS_EUR_kWh': [r['Costo_Unitario_BESS'] for r in final_results],
    'Valore_Attuale_Netto_EUR': [r['VAN'] for r in final_results],
    'Payback_Period_Anni': [r['PBP'] for r in final_results]
})

# Formattazione estetica della tabella per il terminale
tabella_risultati['Valore_Attuale_Netto_EUR'] = tabella_risultati['Valore_Attuale_Netto_EUR'].map('{:,.0f}'.format)
tabella_risultati['Payback_Period_Anni'] = tabella_risultati['Payback_Period_Anni'].apply(
    lambda x: '{:.2f}'.format(x) if np.isfinite(x) else 'Mai (>20 anni)'
)

print(tabella_risultati.to_string(index=False))
print('\n>>> Analisi di Sensitivity completata.')