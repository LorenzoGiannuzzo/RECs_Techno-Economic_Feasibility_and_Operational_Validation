# =========================================================
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# =========================================================
# 1. DATA IMPORT AND PREPARATION
# =========================================================
# Legge il dataset, specificando il separatore e il carattere decimale
df = pd.read_csv('data/data_cs.csv', decimal=',', sep=';')

# Crea una colonna datetime corretta
df['datetime'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Hour'].astype(str),
    format='%d/%m/%Y %H'
)

# =========================================================
# --- NUOVA SEZIONE 1.5: GESTIONE DEI VALORI ZERO ---
# =========================================================
# Conta quanti valori zero sono presenti nella colonna 'E'
zero_count = (df['E'] == 0).sum()
print(f"Trovati {zero_count} valori uguali a zero nella colonna 'E'.")

if zero_count > 0:
    # 1. Sostituisce i valori zero con NaN (Not a Number)
    df.loc[df['E'] == 0, 'E'] = np.nan

    # 2. Applica l'interpolazione lineare per riempire i valori NaN
    df['E'].interpolate(method='linear', inplace=True)

    # 3. Gestisce eventuali valori NaN rimasti all'inizio o alla fine del dataset
    df['E'].fillna(method='ffill', inplace=True)  # Forward-fill
    df['E'].fillna(method='bfill', inplace=True)  # Backward-fill

    print("I valori zero sono stati sostituiti usando l'interpolazione lineare.")
else:
    print("Nessun valore zero da interpolare.")

## =========================================================
# 2. FEATURE ENGINEERING (Solo Variabili Cicliche)
# =========================================================
# Crea feature cicliche per aiutare il modello a capire la periodicità
df['Month'] = df['datetime'].dt.month
df['Hour'] = df['datetime'].dt.hour
df['DayOfWeek'] = df['datetime'].dt.dayofweek

df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 23)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 23)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 6)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 6)

# =========================================================
# 3. TRAIN–TEST SPLIT & NORMALIZZAZIONE (NO LEAKAGE)
# =========================================================
# Definisci le feature (NOTA: 'Test' non è ancora normalizzato) e il target grezzo
input_features = ['Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Test']
output_target = 'E'

X = df[input_features]
y = df[output_target]

# Suddivide i dati (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Recupera i dataframe originali per il plotting futuro
train_df = df.loc[X_train.index].copy()
test_df = df.loc[X_test.index].copy()

# --- CORREZIONE DATA LEAKAGE ---
# Calcola i massimi SOLO sul Train Set
max_E_train = y_train.max()
max_Test_train = X_train['Test'].max()

# Applica la normalizzazione al Train Set
X_train = X_train.copy() # Evita il SettingWithCopyWarning
X_train['Test'] = X_train['Test'] / max_Test_train
y_train_norm = y_train / max_E_train

# Applica la normalizzazione al Test Set (usando i parametri del Train Set!)
X_test = X_test.copy()
X_test['Test'] = X_test['Test'] / max_Test_train
y_test_norm = y_test / max_E_train

# Salva l'intero dataset X normalizzato per la predizione finale (Sezione 10)
X_full_norm = X.copy()
X_full_norm['Test'] = X_full_norm['Test'] / max_Test_train
# =========================================================
# 4. MULTILAYER PERCEPTRON (MLP) REGRESSOR
# =========================================================
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=400,
    random_state=42,
    early_stopping=True
)
# Addestra sui dati normalizzati senza leakage
mlp.fit(X_train, y_train_norm)

train_pred = mlp.predict(X_train)
test_pred = mlp.predict(X_test)

# =========================================================
# 5. ASSIGN PREDICTIONS AND EVALUATE
# =========================================================
train_df['prediction'] = train_pred
test_df['prediction'] = test_pred

# Denormalizza i dati usando IL MASSIMO DEL TRAIN SET
train_df['E_real'] = y_train # Questi sono già i valori reali in kWh
train_df['E_pred'] = train_df['prediction'] * max_E_train

test_df['E_real'] = y_test   # Questi sono già i valori reali in kWh
test_df['E_pred'] = test_df['prediction'] * max_E_train

metrics_train = pd.DataFrame({
    'Metric': ['Global MSE', 'Global MAPE', 'Global R2'],
    'Value': [
        mean_squared_error(y_train_norm, train_pred),
        mean_absolute_percentage_error(y_train_norm, train_pred),
        r2_score(y_train_norm, train_pred)
    ]
})
metrics_test = pd.DataFrame({
    'Metric': ['Global Test MSE', 'Global Test MAPE', 'Global Test R2'],
    'Value': [
        mean_squared_error(y_test_norm, test_pred),
        mean_absolute_percentage_error(y_test_norm, test_pred),
        r2_score(y_test_norm, test_pred)
    ]
})

print("\n--- Training Metrics ---")
print(metrics_train)
print("\n--- Testing Metrics ---")
print(metrics_test)
# =========================================================
# 7. SCATTER PLOT: REAL VS PREDICTED (DENORMALIZED)
# =========================================================
r2_train_plot = r2_score(train_df['E_real'], train_df['E_pred'])
r2_test_plot = r2_score(test_df['E_real'], test_df['E_pred'])

# ---- TRAINING SCATTER PLOT ----
plt.figure(figsize=(7, 6))
plt.scatter(train_df['E_real'], train_df['E_pred'], alpha=0.4)
# La linea rossa rappresenta la predizione perfetta (y = x)
plt.plot([0, df['E'].max()], [0, df['E'].max()], color='red', linewidth=2)
plt.xlabel('Energia Reale [kWh]')
plt.ylabel('Energia Predetta [kWh]')
plt.title(f"Reale vs Predetto – Training Set (R² = {r2_train_plot:.3f})")
plt.grid(True)
plt.tight_layout()
# Salva il grafico in alta qualità (crea prima la cartella se non esiste)
plt.savefig('scatter_train.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ---- TESTING SCATTER PLOT ----
plt.figure(figsize=(7, 6))
plt.scatter(test_df['E_real'], test_df['E_pred'], alpha=0.4, color='green')
# La linea rossa rappresenta la predizione perfetta (y = x)
plt.plot([0, df['E'].max()], [0, df['E'].max()], color='red', linewidth=2)
plt.xlabel('Energia Reale [kWh]')
plt.ylabel('Energia Predetta [kWh]')
plt.title(f"Reale vs Predetto – Testing Set (R² = {r2_test_plot:.3f})")
plt.grid(True)
plt.tight_layout()
# Salva il grafico in alta qualità
plt.savefig('scatter_test.pdf', format='pdf', bbox_inches='tight')
plt.show()

# =========================================================
# 8. I TRE GRAFICI TIME SERIES: REAL VS PREDICTED (DENORMALIZED)
# =========================================================
# Riordina i dati per data prima di plottare la serie storica
train_df_sorted = train_df.sort_values('datetime')
test_df_sorted = test_df.sort_values('datetime')

# ---- 1. TRAINING TIME SERIES (INTERO SET) ----
plt.figure(figsize=(15, 6))
plt.plot(train_df_sorted['datetime'], train_df_sorted['E_real'], label='Energia Reale', linewidth=1)
plt.plot(train_df_sorted['datetime'], train_df_sorted['E_pred'], label='Energia Predetta', linewidth=1, alpha=0.7)
plt.xlabel('Data e Ora')
plt.ylabel('Energia Oraria [kWh]')
plt.title('Serie Storica – Training Set (Completo)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 2. TESTING TIME SERIES (INTERO SET) ----
plt.figure(figsize=(15, 6))
plt.plot(test_df_sorted['datetime'], test_df_sorted['E_real'], label='Energia Reale', linewidth=1)
plt.plot(test_df_sorted['datetime'], test_df_sorted['E_pred'], label='Energia Predetta', linewidth=1, alpha=0.7)
plt.xlabel('Data e Ora')
plt.ylabel('Energia Oraria [kWh]')
plt.title('Serie Storica – Testing Set (Completo)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 3. PLOT DI DETTAGLIO SU UN BREVE PERIODO ----
# Seleziona un sottoinsieme di dati dal test set ordinato (es. una settimana)
plot_subset_df = test_df_sorted.head(24 * 7)  # 7 giorni di dati

plt.figure(figsize=(15, 6))
plt.plot(plot_subset_df['datetime'], plot_subset_df['E_real'], label='Energia Reale', linewidth=2, marker='o', markersize=4)
plt.plot(plot_subset_df['datetime'], plot_subset_df['E_pred'], label='Energia Predetta', linewidth=2, linestyle='--')
plt.xlabel('Data e Ora')
plt.ylabel('Energia Oraria [kWh]')
plt.title('Serie Storica – Dettaglio su una Settimana del Test Set')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================================================
# 9. SHAP VALUES (EXPLAINABILITY)
# =========================================================
# Nota: I valori SHAP sono espressi nella stessa unità del target (kWh)
import shap

print("\nCalcolo dei valori SHAP in corso...")
X_train_summary = shap.kmeans(X_train, 50)
explainer = shap.KernelExplainer(mlp.predict, X_train_summary)
X_test_sampled = X_test.sample(n=300, random_state=42)
shap_values = explainer.shap_values(X_test_sampled)

# =========================================================
# 9. SHAP VALUES & FEATURE IMPORTANCE (EXPLAINABILITY)
# =========================================================
import shap

print("\nCalcolo dei valori SHAP in corso (potrebbe richiedere alcuni minuti)...")

# Per i modelli "black-box" come la MLP di scikit-learn, usiamo il KernelExplainer.
# Per ottimizzare i tempi di calcolo, riassumiamo il set di addestramento in 50 centroidi rappresentativi.
X_train_summary = shap.kmeans(X_train, 50)
explainer = shap.KernelExplainer(mlp.predict, X_train_summary)

# Campioniamo 300 righe dal test set per calcolare i valori SHAP in tempi ragionevoli.
# (300 campioni sono statisticamente più che sufficienti per delineare l'importanza globale)
X_test_sampled = X_test.sample(n=300, random_state=42)

# Calcolo effettivo dei valori SHAP
shap_values = explainer.shap_values(X_test_sampled)

# ---- SHAP SUMMARY PLOT (DOT) ----
# Mostra l'impatto positivo o negativo di ogni variabile sulla predizione
plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot - Impatto e Direzione delle Variabili", fontsize=14, pad=20)
shap.summary_plot(shap_values, X_test_sampled, show=False)
plt.tight_layout()
plt.show()

# ---- SHAP BAR PLOT (FEATURE IMPORTANCE GLOBALE) ----
# Mostra l'importanza media assoluta di ogni variabile (simile alla Gini Importance dell'albero)
plt.figure(figsize=(10, 6))
plt.title("SHAP Bar Plot - Importanza Globale delle Feature", fontsize=14, pad=20)
shap.summary_plot(shap_values, X_test_sampled, plot_type="bar", show=False, color="#1f77b4")
plt.tight_layout()
plt.show()

print("Analisi SHAP completata.")


# =========================================================
# 10. ESPORTAZIONE PREVISIONI RESIDENZIALI SCALATE (PER MILP)
# =========================================================
import os

print("\n--- Generazione del profilo di carico Residenziale Predetto ---")
# 1. Previsione sull'intero anno solare usando le X normalizzate (calcolate nella Sez. 3)
pred_intero_anno_norm = mlp.predict(X_full_norm)

# 2. Denormalizzazione usando il massimo del Train Set
pred_residenziale_23_pod = pred_intero_anno_norm * max_E_train

# 3. Scaling del residenziale (da 23 a 781 famiglie equivalenti)
fattore_scala = 781 / 23  # Circa 33.956
pred_residenziale_scalato = pred_residenziale_23_pod * fattore_scala

print(f"Fattore di scala applicato (781/23): {fattore_scala:.3f}")

# 4. Creazione del DataFrame da esportare
df_export_residenziale = pd.DataFrame({
    'Timestamp': df['datetime'], # Mantiene la data e l'ora corrette
    'Carico_kWh': pred_residenziale_scalato
})

# 5. Salvataggio in Excel nella cartella 'tesi'
os.makedirs('tesi', exist_ok=True) # Crea la cartella se non esiste
nome_file_export = 'tesi/CER_Carico_Residenziale_PREDETTO.xlsx'
df_export_residenziale.to_excel(nome_file_export, index=False)

print(f"Esportazione completata! File salvato in: {nome_file_export}")