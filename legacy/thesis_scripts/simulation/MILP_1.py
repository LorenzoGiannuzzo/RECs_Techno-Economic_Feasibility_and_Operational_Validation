# -*- coding: utf-8 -*-
"""
Script di Ottimizzazione MILP (v3.4 - Finale e Completo)
Questo script calcola la strategia ottima di gestione di un BESS in una CER
per massimizzare i revenues. È parametrico per il numero di batterie e
produce un'analisi grafica e un report Excel dettagliato.
"""
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus
import matplotlib.pyplot as plt

print(">>> Start Ottimizzazione MILP (Modello Stand-Alone)...")

# =============================================================================
# 1. IMPOSTAZIONI E CARICAMENTO DATI
# =============================================================================
# --- Parameters dello Scenario (MODIFICA QUI) ---
n_BESS = 1
pv_plant_power = 1000

# --- Parameters Economici ---
base_incentivo = 60;
cap_incentivo = 100;
correzione_geografica = 0;
arera_fee = 10.57
usa_incentivo_PNRR = False;
fattore_riduzione_PNRR = 0.6
if usa_incentivo_PNRR: print(">>> ATTENZIONE: Applicata riduzione del 40% alla tariff premio per incentive PNRR.")

# Aggiungi questo codice nella SEZIONE 1, sotto il loading degli altri file

# --- Loading Dati di Consumo per Categoria (per figure finale) ---
try:
    df_pa = pd.read_excel("CER_Carico_PA.xlsx")
    df_illuminazione = pd.read_excel("CER_Carico_Illuminazione.xlsx")
    df_residenziali = pd.read_excel("CER_Carico_Residenziali_k.xlsx")
    df_industria = pd.read_excel("CER_Carico_Industria.xlsx")

    # Mettiamo tutti i consumi orari in un unico DataFrame per comodità
    consumi_categorie_df = pd.DataFrame({
        'Pubblica Amm.': df_pa['Carico_kWh'].values,
        'Illuminazione': df_illuminazione['Carico_kWh'].values,
        'Residenziali': df_residenziali['Carico_kWh'].values,
        'Industria': df_industria['Carico_kWh'].values
    })
    print(">>> Dati di consumption per category loaded per il figure di copertura.")
except FileNotFoundError as e:
    print(f"ERRORE: File di consumption per category not found - {e}. Il figure di copertura non potrà essere generato.")
    consumi_categorie_df = None # Impostiamo a None per saltare il figure dopo

# ... il resto della SEZIONE 1 continua ...

# --- Loading e Preparation Dati ---
try:
    df_consumo = pd.read_excel("CER_Carico_Orario_Aggregato_k.xlsx")
    rec_consumption = df_consumo['Carico_kWh'].values
    print(">>> Dati di consumption loaded.")

    df_pz = pd.read_excel("MGP_PrezziZonali_SUD.xlsx", decimal=',', header=0)
    df_pz.columns = ['Data', 'Ora', 'Prezzo_MWh']
    df_pz['Timestamp'] = pd.to_datetime(df_pz['Data'], dayfirst=True) + pd.to_timedelta(df_pz['Ora'] - 1, unit='h')
    df_pz = df_pz.set_index('Timestamp').sort_index()
    df_pz_2025 = df_pz.loc['2025-01-01':'2025-12-31 23:59:59']
    Pz_orario = df_pz_2025['Prezzo_MWh'].values
    print(">>> Dati di Prezzo Zonale Orario loaded.")

    file_produzione = "local_data/CaseStudy_Project_VC0_HourlyRes_0.CSV"
    df_produzione = pd.read_csv(file_produzione, sep=';', skiprows=10, header=None, usecols=[1],
                                names=['Produzione_str'], dtype=str)
    pv_production = pd.to_numeric(df_produzione['Produzione_str'].str.replace(',', '.'), errors='coerce').fillna(
        0).values[:8760]
    print(">>> Dati di production FV loaded e puliti.")
except FileNotFoundError as e:
    print(f"ERRORE: File not found - {e}. Controlla i percorsi e i nomi dei file.");
    exit()

# --- Parameters BESS (calcolati dinamicamente) ---
E_nom_singolo = 760;
P_charge_singolo = 380;
P_discharge_singolo = 380
E_nom = E_nom_singolo * n_BESS;
P_charge_max = P_charge_singolo * n_BESS;
P_discharge_max = P_discharge_singolo * n_BESS
scenario_txt = f'Scenario: {pv_plant_power} kWp con {n_BESS}x BESS ({E_nom}kWh/{P_charge_max}kW)'
if usa_incentivo_PNRR: scenario_txt += ' con incentive PNRR'
print(f'>>> {scenario_txt.upper()}')
eta_BESS = 0.95;
SoC_min = 0.10;
SoC_max = 1.00;
SoC_initial = 0.5
E_min = E_nom * SoC_min;
E_max = E_nom * SoC_max;
eta_cycle = eta_BESS ** 0.5
print(f'>>> Parameters BESS totali: E={E_nom}kWh, P_charge/discharge={P_charge_max}kW')

n_hours = 8760;
hours = range(n_hours)


# =============================================================================
# 2. DEFINIZIONE DEL PROBLEMA DI OTTIMIZZAZIONE
# =============================================================================
prob = LpProblem("Massimizzazione_Ricavi_CER_StandAlone", LpMaximize)

E_batt = LpVariable.dicts("E", hours, lowBound=E_min, upBound=E_max)
P_carica_da_FV = LpVariable.dicts("Pc", hours, lowBound=0, upBound=P_charge_max)
P_scarica = LpVariable.dicts("Pd", hours, lowBound=0, upBound=P_discharge_max)
P_immissione_da_FV = LpVariable.dicts("Pifv", hours, lowBound=0)
P_immissione_da_BESS = LpVariable.dicts("Pibess", hours, lowBound=0)
P_prelievo = LpVariable.dicts("Pp", hours, lowBound=0)
E_shared = LpVariable.dicts("Ec", hours, lowBound=0)
is_charging = LpVariable.dicts("d", hours, cat=LpBinary)

hourly_incentive_tariff = []
for h in hours:
    parte_variabile = max(0, 180 - Pz_orario[h]);
    base_tariff = base_incentivo + parte_variabile
    computed_tariff = min(cap_incentivo, base_tariff + correzione_geografica)
    hourly_incentive_tariff.append(
        computed_tariff * fattore_riduzione_PNRR if usa_incentivo_PNRR else computed_tariff)

prob += lpSum(
    (E_shared[h] / 1000) * (hourly_incentive_tariff[h] + arera_fee) +
    ((P_immissione_da_FV[h] + P_immissione_da_BESS[h]) / 1000) * Pz_orario[h]
    for h in hours
), "Ricavo_Totale_Annuo"

# =============================================================================
# 3. DEFINIZIONE DEI VINCOLI
# =============================================================================
E_batt_iniziale = E_nom * SoC_initial
for h in hours:
    prob += pv_production[h] == P_carica_da_FV[h] + P_immissione_da_FV[h], f"Bilancio_FV_{h}"
    prob += P_scarica[h] == P_immissione_da_BESS[h], f"Bilancio_Immissione_BESS_{h}"
    prob += rec_consumption[h] == E_shared[h] + P_prelievo[h], f"Bilancio_CER_{h}"
    if h == 0:
        prob += E_batt[h] == E_batt_iniziale + P_carica_da_FV[h] * eta_cycle - P_scarica[
            h] / eta_cycle, f"Batteria_ora_{h}"
    else:
        prob += E_batt[h] == E_batt[h - 1] + P_carica_da_FV[h] * eta_cycle - P_scarica[
            h] / eta_cycle, f"Batteria_ora_{h}"
    prob += P_carica_da_FV[h] <= P_charge_max * is_charging[h], f"Logica_Carica_{h}"
    prob += P_scarica[h] <= P_discharge_max * (1 - is_charging[h]), f"Logica_Scarica_{h}"
    total_injection = P_immissione_da_FV[h] + P_immissione_da_BESS[h]
    prob += E_shared[h] <= rec_consumption[h], f"Condivisa_Limite_Consumo_{h}"
    prob += E_shared[h] <= total_injection, f"Condivisa_Limite_Immissione_{h}"

print(">>> Modello MILP (Stand-Alone) definito. Start la risoluzione...")

# =============================================================================
# 4. RISOLUZIONE E ANALISI RISULTATI
# =============================================================================
prob.solve()
print(f">>> Ottimizzazione completata. Stato: {LpStatus[prob.status]}")

if prob.status == 1:
    results = pd.DataFrame({
        'Consumo_CER_kWh': rec_consumption, 'Produzione_FV_kWh': pv_production, 'Pz_Eur_MWh': Pz_orario,
        'Energia_Condivisa_kWh': [E_shared[h].varValue for h in hours],
        'Prelievo_Rete_kWh': [P_prelievo[h].varValue for h in hours],
        'Immissione_Rete_kWh': [P_immissione_da_FV[h].varValue + P_immissione_da_BESS[h].varValue for h in hours],
        'Carica_Batteria_kWh': [P_carica_da_FV[h].varValue for h in hours],
        'Scarica_Batteria_kWh': [P_scarica[h].varValue for h in hours],
        'SoC_Batteria_Percentuale': [E_batt[h].varValue / E_nom * 100 if E_nom > 0 else 0 for h in hours],
    })
else:
    print(">>> L'ottimizzazione non ha trovato una soluzione ottimale."); exit()

# =============================================================================
# 5. CREAZIONE GRAFICI DI ANALISI
# =============================================================================
print(">>> Creation figures di analisi...")


def create_heatmap(ax, data, title, cmap='plasma'):
    heatmap_data = data.values.reshape(365, 24).T
    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, interpolation='gaussian')
    ax.set_title(title);
    ax.set_xlabel('Giorno dell''Anno');
    ax.set_ylabel('Ora del Giorno')
    plt.colorbar(im, ax=ax, label='Potenza (kW) o Prezzo (€/MWh)')


fig1, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
fig1.suptitle(f'Analisi Operativa Annuale Ottimizzata - {scenario_txt}', fontsize=16, fontweight='bold')
create_heatmap(axes[0, 0], results['Produzione_FV_kWh'], 'Produzione FV (kW)')
create_heatmap(axes[0, 1], pd.Series(Pz_orario), 'Prezzo Zonale Orario (€/MWh)')
create_heatmap(axes[1, 0], results['Carica_Batteria_kWh'], 'Carica Batteria (kW)')
create_heatmap(axes[1, 1], results['Scarica_Batteria_kWh'], 'Scarica Batteria (kW)')
create_heatmap(axes[2, 0], results['Immissione_Rete_kWh'], 'Immissione in Rete (kW)')
axes[2, 1].axis('off');
plt.show();
print(">>> Figura 1 (Heatmaps) creata.")

results['Timestamp'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(results.index, unit='h')
results['Mese'] = results['Timestamp'].dt.month
results['Weekday'] = results['Timestamp'].dt.weekday
is_weekday = (results['Weekday'] <= 4);
isFestivo = ~is_weekday
stagioni_map = {m: 'Inverno' for m in [12, 1, 2]};
stagioni_map.update({m: 'Primavera' for m in [3, 4, 5]});
stagioni_map.update({m: 'Estate' for m in [6, 7, 8]});
stagioni_map.update({m: 'Autunno' for m in [9, 10, 11]})
results['Stagione'] = results['Mese'].map(stagioni_map)
fig2, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True);
fig2.suptitle(f'Profilo di Consumo e Copertura Ottimizzati - {scenario_txt}', fontsize=16, fontweight='bold')
stagioni = ['Inverno', 'Primavera', 'Estate', 'Autunno']
for i, stagione in enumerate(stagioni):
    for k, tipo_giorno in enumerate(['Feriale', 'Festivo']):
        ax1 = axes[k, i];
        df_stag = results[
            (is_weekday if tipo_giorno == 'Feriale' else isFestivo) & (results['Stagione'] == stagione)]
        if not df_stag.empty:
            profilo = df_stag.groupby(df_stag['Timestamp'].dt.hour).mean(numeric_only=True)
            ax1.fill_between(profilo.index, profilo['Energia_Condivisa_kWh'], color=[0.2, 0.8, 0.2], alpha=0.7,
                             label='Energia Condivisa')
            ax1.plot(profilo.index, profilo['Consumo_CER_kWh'], 'r-', linewidth=2.5, label='Consumo Aggregato CER')
            ax1.plot(profilo.index, profilo['Immissione_Rete_kWh'], color=[1, 0.6, 0], linestyle=':', linewidth=2,
                     label='Energia tot. Immessa (per RID)')
            ax1.set_title(f'{stagione} - {tipo_giorno}');
            ax1.set_ylabel('Energia (kWh)');
            ax1.grid(True);
            ax1.set_xticks(range(0, 24, 4));
            ax1.set_xlim(0, 23)
            ax2 = ax1.twinx();
            ax2.plot(profilo.index, profilo['SoC_Batteria_Percentuale'], 'm--.', linewidth=2,
                     label='SOC Medio Batteria (%)')
            ax2.set_ylabel('Stato di Carica (%)', color='m');
            ax2.tick_params(axis='y', labelcolor='m');
            ax2.set_ylim(0, 100)
            if i == 0 and k == 0: lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax2.legend(
                lines + lines2, labels + labels2, loc='upper left')
            if k == 1: ax1.set_xlabel('Ora del Giorno')
plt.show();
print(">>> Figura 2 (Figures Stagionali) creata.")





# Creation della figura 1x4 (4 stagioni)
fig3, axes = plt.subplots(1, 4, figsize=(18, 6))

# Aggiunge spazio in basso per poter inserire la legenda esterna
plt.subplots_adjust(bottom=0.25, wspace=0.3)

fig3.suptitle(f'Profili medi stagionali - (MILP)', fontsize=16, fontweight='bold')
stagioni = ['Inverno', 'Primavera', 'Estate', 'Autunno']

lines_all = []
labels_all = []

for i, stagione in enumerate(stagioni):
    ax1 = axes[i]

    # Filtriamo i results solo per la stagione (nessuna distinzione feriale/festivo)
    df_stag = results[results['Stagione'] == stagione]

    if not df_stag.empty:
        # Computation del profilo hourly medio
        profilo = df_stag.groupby(df_stag['Timestamp'].dt.hour).mean(numeric_only=True)

        # Tracciamento data sull'asse y principale
        ax1.fill_between(profilo.index, profilo['Energia_Condivisa_kWh'], color=[0.2, 0.8, 0.2], alpha=0.7,
                         label='Energia shared (autoconsumata)')
        ax1.plot(profilo.index, profilo['Consumo_CER_kWh'], 'r-', linewidth=2.5, label='Consumo aggregato CER')
        ax1.plot(profilo.index, profilo['Immissione_Rete_kWh'], color=[1, 0.6, 0], linestyle=':', linewidth=2,
                 label='Energia tot. immessa (per RID)')

        # Settings figure principale
        ax1.set_title(f'{stagione}', fontsize=14)
        ax1.set_ylabel('Energia (kWh)')
        ax1.set_xlabel('Ora del Giorno')
        ax1.grid(True)
        ax1.set_xticks(range(0, 24, 4))
        ax1.set_xlim(0, 23)

        # Tracciamento data sull'asse y secondario (SoC Batteria)
        ax2 = ax1.twinx()
        ax2.plot(profilo.index, profilo['SoC_Batteria_Percentuale'], 'm--.', linewidth=2,
                 label='SOC Medio Batteria (%)')

        ax2.set_ylabel('Stato di Carica (%)', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        ax2.set_ylim(0, 100)

        # Raccogliamo le info della legenda solo dal primo figure per non ripeterla
        if i == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines_all = lines1 + lines2
            labels_all = labels1 + labels2

# Inserimento della legenda globale unica sotto i figures
fig3.legend(lines_all, labels_all, loc='upper center', bbox_to_anchor=(0.5, 0.12),
            ncol=4, fontsize=12, frameon=True, borderpad=1)

plt.show()
print(">>> Nuovo figure stagionale (unificato) con legenda esterna creato.")
# =============================================================================
# 5.5. CREAZIONE GRAFICO COPERTURA FABBISOGNO PER CATEGORIA
# =============================================================================
# Controlliamo se i data delle categories sono stati loaded correttamente
if consumi_categorie_df is not None and prob.status == 1:
    print(">>> Creation figure di copertura del fabbisogno per category...")

    # --- Computation della Ripartizione dell'Energia Condivisa ---
    consumi_categorie_df['Totale_Orario'] = consumi_categorie_df.sum(axis=1)
    frazioni_consumo_df = consumi_categorie_df.iloc[:, :-1].div(consumi_categorie_df['Totale_Orario'], axis=0).fillna(0)
    shared_energy_per_category = frazioni_consumo_df.multiply(results['Energia_Condivisa_kWh'], axis=0)

    consumo_annuo_per_categoria = consumi_categorie_df.sum(axis=0)
    annual_shared_per_category = shared_energy_per_category.sum(axis=0)

    valori_condivisa = annual_shared_per_category.values
    valori_prelevata = (consumo_annuo_per_categoria.iloc[:-1] - annual_shared_per_category).values
    categories = consumi_categorie_df.columns[:-1]

    # --- Creation del Figure ---
    fig3, ax = plt.subplots(figsize=(12, 7))

    ax.bar(categories, valori_condivisa, label='Energia Condivisa Virtualmente', color='#2ca02c', edgecolor='black')
    ax.bar(categories, valori_prelevata, bottom=valori_condivisa, label='Non Condivisa', color='#d62728',
           edgecolor='black')

    # --- Aggiunta delle Percentuali di Copertura ---
    consumo_totale_cat = valori_condivisa + valori_prelevata
    for i, total in enumerate(consumo_totale_cat):
        if total > 0:
            percentuale_copertura = (valori_condivisa[i] / total) * 100
            ax.text(i, total + 50000, f'{percentuale_copertura:.1f}%', ha='center', va='bottom', fontweight='bold')

    # --- Stile e Titoli ---
    ax.set_ylabel('Consumo Energetico Annuo (kWh)', fontsize=12)

    # --- RIGA MODIFICATA PER IL NUOVO TITOLO ---
    titolo_scenario_milp = scenario_txt.replace("Scenario:", "Scenario MILP:")
    ax.set_title(f'Quota di Consumo Incentivabile per Categoria di Utenza\n{titolo_scenario_milp}', fontsize=14,
                 fontweight='bold')
    # -------------------------------------------

    ax.legend()
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, axis='y')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))  # Mostra y-axis in milioni (x 10^6)

    plt.tight_layout()
    plt.show()
    print(">>> Figura 3 (Copertura Fabbisogno) creata.")

# =============================================================================
# 6. ANALISI ECONOMICA FINALE ED ESPORTAZIONE
# =============================================================================
print('\n--- ANALISI ECONOMICA E ESPORTAZIONE (Modello Stand-Alone) ---\n')

# --- Computation dei Ricavi Orari (già disponibili dall'ottimizzazione, ma li ricalcoliamo per chiarezza) ---
results['Tariffa_Incentivo_Oraria_Eur_MWh'] = hourly_incentive_tariff
results['Ricavo_Incentivo_Orario_Eur'] = (results['Energia_Condivisa_kWh'] / 1000) * results[
    'Tariffa_Incentivo_Oraria_Eur_MWh']
results['Ricavo_ARERA_Orario_Eur'] = (results['Energia_Condivisa_kWh'] / 1000) * arera_fee
results['Ricavo_RID_Orario_Eur'] = (results['Immissione_Rete_kWh'] / 1000) * results['Pz_Eur_MWh']

# --- Computation e Stampa dei Totali Annuali ---
annual_shared_energy = results['Energia_Condivisa_kWh'].sum() / 1000
annual_injected_energy = results['Immissione_Rete_kWh'].sum() / 1000
annual_incentive_revenue = results['Ricavo_Incentivo_Orario_Eur'].sum()
annual_arera_revenue = results['Ricavo_ARERA_Orario_Eur'].sum()
annual_rid_revenue = results['Ricavo_RID_Orario_Eur'].sum()
annual_total_revenue = annual_incentive_revenue + annual_arera_revenue + annual_rid_revenue

print(f"Energia Condivisa Annua (per incentive e ARERA): {annual_shared_energy:.2f} MWh")
print(f"Energia Immessa Annua (per RID):                 {annual_injected_energy:.2f} MWh\n")
print(f"Dettaglio Ricavi Annuali Stimati:")
print(f"  - Incentivo Tariffa CACER:                  € {annual_incentive_revenue:,.2f}")
print(f"  - Corrispettivo ARERA:                       € {annual_arera_revenue:,.2f}")
print(f"  - Ritiro Dedicato (RID):                      € {annual_rid_revenue:,.2f}")
print("--------------------------------------------------")
print(f"RICAVO TOTALE ANNUO OTTIMO: € {annual_total_revenue:,.2f}")
print("--------------------------------------------------")

# --- Saving in Excel con tutte le colonne ---
nome_file_excel = f"Results_Ottimizzazione_StandAlone_{n_BESS}xBESS.xlsx"
colonne_da_salvare = [
    'Timestamp', 'Consumo_CER_kWh', 'Produzione_FV_kWh', 'Energia_Condivisa_kWh',
    'Prelievo_Rete_kWh', 'Immissione_Rete_kWh', 'Carica_Batteria_kWh', 'Scarica_Batteria_kWh',
    'SoC_Batteria_Percentuale', 'Pz_Eur_MWh', 'Tariffa_Incentivo_Oraria_Eur_MWh',
    'Ricavo_Incentivo_Orario_Eur', 'Ricavo_ARERA_Orario_Eur', 'Ricavo_RID_Orario_Eur'
]
final_results = results[colonne_da_salvare].set_index('Timestamp')

try:
    final_results.to_excel(nome_file_excel, sheet_name='Dati Orari')
    print(f'>>> Successo! File Excel "{nome_file_excel}" creato successfully.')
except Exception as e:
    print(f'>>> ERRORE durante la scrittura del file Excel: {e}')


print('>>> Analisi completata.')

# Aggiungi questo alla end della SEZIONE 6 del tuo "Script MILP"
# -----------------------------------------------------------------------------
# 7. SALVATAGGIO RISULTATO PER ANALISI FINANZIARIA
# -----------------------------------------------------------------------------
# Importa la libreria necessaria (se non l'hai già fatto all'start)
from scipy.io import savemat

# Definisci il nome del file .mat in modo che sia coerente con gli altri
nome_file_mat = f"Dati_OTTIMIZZAZIONE_{n_BESS}xBESS.mat"

# Crea un dizionario con la variabile da salvare (il nome deve essere 'annual_total_revenue')
data_to_save = {'annual_rid_revenue': annual_rid_revenue}

# Salva il file .mat
savemat(nome_file_mat, data_to_save)

print(f'\n>>> Risultato per analisi finanziaria salvato in "{nome_file_mat}"')
# -----------------------------------------------------------------------------
