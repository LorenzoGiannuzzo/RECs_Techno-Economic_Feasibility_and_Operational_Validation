import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# -------------------------------------------------------------------------
# SCRIPT PER L'ANALISI FINANZIARIA DELLA CER (v2.0 - Basato su Ricavi RID)
# -------------------------------------------------------------------------
# Calcola PBP e VAN considerando come unico revenue la vendita di energy (RID),
# per valutare la sostenibilità dell'investment dell'impianto.
# -------------------------------------------------------------------------

# %% 1. IMPOSTAZIONI E DEFINIZIONE DEGLI SCENARI
print('>>> Start Analisi Finanziaria Comparativa (basata su RID)...\n')

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

# Parameters finanziari
pv_plant_power_kW = 1000
costo_unitario_FV = 800
costo_unitario_BESS = 350
capacita_BESS_singolo = 760
OPEX_percentuale = 1.5
contributo_PNRR_percentuale = 0
durata_analisi = 20
tasso_attualizzazione = 4
final_results = []

# %% 2. CICLO DI CALCOLO PER OGNI SCENARIO
for i, current_scenario in enumerate(scenarios):
    print(f"--- Elaborazione Scenario {i + 1}: {current_scenario['Nome']} ---")
    try:
        percorso_completo = os.path.join(cartella_file, current_scenario['FileMAT'])
        loaded_data = loadmat(percorso_completo)

        # --- MODIFICA CHIAVE EFFETTUATA ---
        # Il revenue annual è hour basato solo sul Ritiro Dedicato (RID).
        annual_revenue = loaded_data['annual_rid_revenue'][0][0]

    except FileNotFoundError:
        print(f"ATTENzione: Impossibile trovare il file {percorso_completo}. Salto lo scenario.")
        continue
    except KeyError:
        print(
            f"ATTENZIONE: La variabile 'annual_rid_revenue' non trovata in {current_scenario['FileMAT']}. Salto lo scenario.")
        continue

    # Computation CAPEX
    CAPEX_FV = pv_plant_power_kW * costo_unitario_FV
    CAPEX_BESS = current_scenario['NumBESS'] * capacita_BESS_singolo * costo_unitario_BESS
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
        'VAN': VAN,
        'PBP': PBP,
        'FlussoCumulato': cumulative_discounted_flow,
        'Colore': current_scenario['Colore'],
        'Stile': current_scenario['Stile']
    })
    print('Calcoli completati.\n')

# %% 3. CREAZIONE GRAFICO COMPARATIVO DEL PAYBACK PERIOD
fig, ax = plt.subplots(figsize=(12, 8))
for ris in final_results:
    ax.plot(range(durata_analisi + 1), ris['FlussoCumulato'], ris['Stile'],
            linewidth=2, color=ris['Colore'],
            markerfacecolor=ris['Colore'], markersize=5,
            label=ris['Nome'])
ax.axhline(0, color='k', linewidth=1.5)
for ris in final_results:
    if np.isfinite(ris['PBP']):
        ax.axvline(x=ris['PBP'], color=ris['Colore'], linestyle='--', linewidth=1.5)
        ax.plot(ris['PBP'], 0, 'o', markersize=8, markerfacecolor=ris['Colore'], markeredgecolor='k')
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min - y_max * 0.05, y_max * 1.05)
ax.grid(True)
ax.set_title('Analisi PBP basata su Ricavi da Vendita Energia (RID)', fontsize=14, fontweight='bold')
ax.set_xlabel('Anni', fontsize=12)
ax.set_ylabel('Flusso di Cassa Cumulato Attualizzato (€)', fontsize=12)
ax.legend(loc='lower right')
ax.set_xticks(np.arange(0, durata_analisi + 1, 2))
plt.tight_layout()
plt.show()
print('>>> Figure comparativo del Payback Period creato.')

# %% 4. STAMPA TABELLA RIASSUNTIVA FINALE
print('\n\n--- TABELLA RIASSUNTIVA DEGLI INDICATORI FINANZIARI DELL\'IMPIANTO ---')
nomi_scenari = [r['Nome'] for r in final_results]
van_valori = [r['VAN'] for r in final_results]
pbp_valori = [r['PBP'] for r in final_results]
tabella_risultati = pd.DataFrame({
    'Scenario': nomi_scenari,
    'Valore_Attuale_Netto_EUR': van_valori,
    'Payback_Period_Anni': pbp_valori
})
tabella_risultati['Valore_Attuale_Netto_EUR'] = tabella_risultati['Valore_Attuale_Netto_EUR'].map('{:,.0f}'.format)
tabella_risultati['Payback_Period_Anni'] = tabella_risultati['Payback_Period_Anni'].map('{:.2f}'.format)
print(tabella_risultati.to_string(index=False))
print('\n>>> Analisi completata.')