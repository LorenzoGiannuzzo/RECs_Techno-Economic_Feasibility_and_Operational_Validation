import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. IMPOSTAZIONI E CONFIGURAZIONE AVANZATA
# =============================================================================
print(">>> Start analisi finale di ripartizione dei guadagni (v4.1 - Corretto)...")

# --- FILE E SCENARIO ---
file_risultati_ottimizzazione = "Results_Ottimizzazione_StandAlone_1xBESS.xlsx"

# --- PARAMETRI DELLA CER (DA STATUTO) ---
quota_trattenuta_cer_percentuale = 10.0

# --- PARAMETRI NORMATIVI PER TARIFFA ECCEDENTARIA ---
ha_ricevuto_contributi_pnrr = False
soglia_percentuale_standard = 55.0
soglia_percentuale_pnrr = 45.0

# --- PARAMETRI DEI MEMBRI (POD) ---
file_consumi_pods = {
    'Pubblica Amm.': 'CER_Carico_PA.xlsx',
    'Illuminazione': 'CER_Carico_Illuminazione.xlsx',
    'Residenziali': 'CER_Carico_Residenziali_k.xlsx',
    'Industria': 'CER_Carico_Industria.xlsx'
}
pods_impresa = ['Industria']

# --- PARAMETRI SCALA UTENZE RESIDENZIALI ---
numero_famiglie_campione = 23
fattore_scala_k = 33.96
numero_totale_famiglie = int(numero_famiglie_campione * fattore_scala_k)

# =============================================================================
# 2. CARICAMENTO DATI
# =============================================================================
try:
    df_risultati_cer = pd.read_excel(file_risultati_ottimizzazione, index_col='Timestamp')
    print(f">>> File dei results '{file_risultati_ottimizzazione}' caricato.")
    consumi_pods = {k: pd.read_excel(v, index_col='Timestamp')['Carico_kWh'] for k, v in file_consumi_pods.items()}
    df_consumi_pods = pd.DataFrame(consumi_pods)
    print(">>> Profili di consumption per ogni POD loaded.")
except FileNotFoundError as e:
    print(f"ERRORE: File not found - {e}.")
    exit()

# =============================================================================
# 3. ANALISI DEI RICAVI TOTALI DELLA CER
# =============================================================================
print("\n" + "=" * 50)
print("FASE 1: ANALISI DEI RICAVI TOTALI DELLA CER")
print("=" * 50)

# --- 3.1 Calcoli annuali totali ---
annual_shared_energy = df_risultati_cer['Energia_Condivisa_kWh'].sum()
annual_injected_energy = df_risultati_cer['Immissione_Rete_kWh'].sum()
annual_incentive_revenue = df_risultati_cer['Ricavo_Incentivo_Orario_Eur'].sum()
annual_arera_revenue = df_risultati_cer['Ricavo_ARERA_Orario_Eur'].sum()
annual_rid_revenue = df_risultati_cer['Ricavo_RID_Orario_Eur'].sum()
ricavo_totale_cer = annual_incentive_revenue + annual_arera_revenue + annual_rid_revenue

# --- 3.2 Stampa del sommario dei revenues totali ---
print(f"\nSOMMARIO RICAVI ANNUALI LORDI GENERATI DALLA CER:")
print(f"  - Ricavo da Incentivo Tariffa Premio: € {annual_incentive_revenue:,.2f}")
print(f"  - Ricavo da Corrispettivo ARERA:      € {annual_arera_revenue:,.2f}")
print(f"  - Ricavo da Ritiro Dedicato (RID):    € {annual_rid_revenue:,.2f}")
print("-" * 40)
print(f"  RICAVO TOTALE ANNUALE:                € {ricavo_totale_cer:,.2f}\n")

# --- 3.3 Figure a torta dei revenues totali ---
labels = ['Incentivo Tariffa Premio', 'Corrispettivo ARERA', 'Ritiro Dedicato (RID)']
sizes = [annual_incentive_revenue, annual_arera_revenue, annual_rid_revenue]
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85,
        colors=['#4daf4a', '#377eb8', '#e41a1c'])
ax1.axis('equal')
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig1.gca().add_artist(centre_circle)
plt.title('Composizione dei Ricavi Totali Annuali della CER', fontsize=16, fontweight='bold')
plt.show()

# =============================================================================
# 4. SCOMPOSIZIONE DEGLI INCENTIVI DISTRIBUIBILI
# =============================================================================
print("\n" + "=" * 50)
print("FASE 2: SCOMPOSIZIONE DEGLI INCENTIVI")
print("=" * 50)

# --- 4.1 Computation quote ---
totale_incentivi_lordi = annual_incentive_revenue + annual_arera_revenue
quota_trattenuta_valore = totale_incentivi_lordi * (quota_trattenuta_cer_percentuale / 100)
incentivi_netti_da_distribuire = totale_incentivi_lordi - quota_trattenuta_valore

# --- 4.2 Gestione Tariffa Premio Eccedentaria ---
soglia_attuale = soglia_percentuale_pnrr if ha_ricevuto_contributi_pnrr else soglia_percentuale_standard
rapporto_condivisa_immessa = (annual_shared_energy / annual_injected_energy) * 100 if annual_injected_energy > 0 else 0
surplus_incentive = 0
ordinary_incentive = incentivi_netti_da_distribuire

print(f"\nTotale incentivi lordi (Incentivo+ARERA): € {totale_incentivi_lordi:,.2f}")
print(f"Quota trattenuta dalla CER ({quota_trattenuta_cer_percentuale}%): € {quota_trattenuta_valore:,.2f}")
print(f"Totale incentivi netti da distribuire ai membri: € {incentivi_netti_da_distribuire:,.2f}")
print("-" * 40)
print(f"Rapporto Energia Condivisa/Immessa: {rapporto_condivisa_immessa:.2f}% (Soglia applicata: {soglia_attuale}%)")

if rapporto_condivisa_immessa > soglia_attuale:
    print(">>> Il rapporto supera la soglia. L'incentive netto viene scomposto:")
    percentuale_eccedenza = (rapporto_condivisa_immessa - soglia_attuale) / rapporto_condivisa_immessa
    surplus_incentive = incentivi_netti_da_distribuire * percentuale_eccedenza
    ordinary_incentive = incentivi_netti_da_distribuire - surplus_incentive
    print(f"  - QUOTA ORDINARIA (<{soglia_attuale}%): € {ordinary_incentive:,.2f} (distribuita a tutti)")
    print(
        f"  - QUOTA ECCEDENTARIA (>{soglia_attuale}%): € {surplus_incentive:,.2f} (distribuita solo ai non-impresa)")
else:
    print(">>> Il rapporto NON supera la soglia. L'intero incentive netto è considerato ordinario.")

# --- 4.3 Figure di scomposizione ---
if surplus_incentive > 0:
    labels_incentivo = ['Quota Ordinaria', 'Quota Eccedentaria']
    sizes_incentivo = [ordinary_incentive, surplus_incentive]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.bar(labels_incentivo, sizes_incentivo, color=['#3182bd', '#9ecae1'], edgecolor='black')
    ax2.set_ylabel('Valore [€]')
    ax2.set_title('Scomposizione Incentivo Netto Distribuibile', fontsize=16, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.01, f'€ {yval:,.0f}', ha='center', va='bottom')
    plt.show()

# =============================================================================
# 5. RIPARTIZIONE FINALE TRA I MEMBRI E VISUALIZZAZIONE
# =============================================================================
print("\n" + "=" * 50)
print("FASE 3: RIPARTIZIONE FINALE TRA I MEMBRI")
print("=" * 50)

# --- 5.1 Ripartizione (logica invariata) ---
consumi_annuali_pods = df_consumi_pods.sum()
proporzioni_annuali_ordinarie = consumi_annuali_pods / consumi_annuali_pods.sum()
guadagno_ordinario_per_pod = proporzioni_annuali_ordinarie * ordinary_incentive
pods_non_impresa = [p for p in df_consumi_pods.columns if p not in pods_impresa]
consumi_annuali_non_impresa = consumi_annuali_pods[pods_non_impresa].sum()
guadagno_eccedentario_per_pod = pd.Series(0.0, index=df_consumi_pods.columns)

# --- RIGA CORRETTA ---
if consumi_annuali_non_impresa > 0:
    proporzioni_annuali_eccedentarie = consumi_annuali_pods[pods_non_impresa] / consumi_annuali_non_impresa
    guadagno_eccedentario_per_pod.update(proporzioni_annuali_eccedentarie * surplus_incentive)

df_guadagni_finali = pd.DataFrame({'Quota Ordinaria [€]': guadagno_ordinario_per_pod})
df_guadagni_finali['Quota Eccedentaria [€]'] = guadagno_eccedentario_per_pod
df_guadagni_finali['Guadagno Totale Membro [€]'] = df_guadagni_finali.sum(axis=1)
df_guadagni_finali = df_guadagni_finali[
    ['Guadagno Totale Membro [€]', 'Quota Ordinaria [€]', 'Quota Eccedentaria [€]']].fillna(0)

# --- 5.2 Sommario finale e calcoli per famiglia ---
print("\nTABELLA RIASSUNTIVA DEI GUADAGNI ANNUALI DISTRIBUITI AI MEMBRI:")
print(df_guadagni_finali.round(2).to_string())
guadagno_totale_residenziali = df_guadagni_finali.loc['Residenziali', 'Guadagno Totale Membro [€]']
guadagno_per_famiglia = guadagno_totale_residenziali / numero_totale_famiglie
print(f"\nANALISI UTENZE RESIDENZIALI:")
print(
    f"Il gain totale di € {guadagno_totale_residenziali:,.2f} per il POD 'Residenziali' è ripartito tra {numero_totale_famiglie} famiglie.")
print(f"GUADAGNO MEDIO ANNUALE PER FAMIGLIA: € {guadagno_per_famiglia:,.2f}\n")

# --- 5.3 Figure finale ---
print(">>> Creation del figure di ripartizione finale...")
df_plot = df_guadagni_finali[['Quota Ordinaria [€]', 'Quota Eccedentaria [€]']]
fig3, ax3 = plt.subplots(figsize=(12, 8))
df_plot.plot(kind='bar', stacked=True, ax=ax3, color=['#3182bd', '#9ecae1'], edgecolor='black')
for i, total in enumerate(df_guadagni_finali['Guadagno Totale Membro [€]']):
    if total > 0:
        ax3.text(i, total * 1.02, f'€ {total:,.0f}', ha='center', fontweight='bold')
ax3.set_title('Ripartizione Finale Guadagni tra i Membri', fontsize=16, fontweight='bold')
ax3.set_ylabel('Guadagno Annuo [€]')
ax3.set_xlabel('Membro della CER (POD)')
plt.xticks(rotation=0)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.legend(title='Componenti di Guadagno')
footer_text = (
    f"Nota: Il revenue RID (€ {annual_rid_revenue:,.0f}) e una quota trattenuta (€ {quota_trattenuta_valore:,.0f}) rimangono alla CER.\n"
    f"Il gain medio per le {numero_totale_famiglie} famiglie residential è di € {guadagno_per_famiglia:,.2f}/anno."
)
fig3.text(0.5, -0.05, footer_text, ha='center', fontsize=10, style='italic', wrap=True)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()

print(">>> Analisi completata.")