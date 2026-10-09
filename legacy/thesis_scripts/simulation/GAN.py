import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. IMPOSTAZIONI INIZIALI
# =============================================================================
BASE_DATA_DIR = "local_data"
TOTALE_FAMIGLIE_COMUNE = 1116
PERCENTUALE_ADESIONE = 0.3
TARGET_FAMIGLIE_CER = int(round(TOTALE_FAMIGLIE_COMUNE * PERCENTUALE_ADESIONE))

POD_PA = list(range(1, 9))  # 1 a 8
POD_ILLUM = list(range(9, 33))  # 9 a 32
POD_RESID = list(range(33, 56))  # 33 a 55 (23 POD)
POD_INDUS = [56]  # 56

# Parameters GAN
LATENT_DIM = 15  # Dimensione del rumore in ingresso
EPOCHS = 500  # Epoche di addestramento
BATCH_SIZE = 64
LR = 0.0002  # Learning rate

print(f">>> Start script. Target adesione: {TARGET_FAMIGLIE_CER} famiglie.")


# =============================================================================
# 2. DEFINIZIONE DELLE RETI GAN (PyTorch)
# =============================================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 24),
            nn.ReLU()  # ReLU finale perché i consumi (kWh) non possono essere negativi
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(24, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Restituisce probabilità (0 = Falso, 1 = Vero)
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# 3. CARICAMENTO DATI (Simulato/Strutturato come il tuo MATLAB)
# =============================================================================
# NOTA: Per brevità, assumiamo di aver già estratto e allineato i data in una
# matrice NumPy o DataFrame Pandas di dimensioni (8760, 56) come nel tuo MATLAB.
# Qui creo un DataFrame placeholder per permetterti di testare la logica GAN.
# Sostituisci questo blocco con il parsing effettivo dei tuoi CSV tramite pandas.read_csv.

dates_2025 = pd.date_range(start='2025-01-01', end='2025-12-31 23:00:00', freq='1h')
# Dummy data: sostituisci con la tua matrice real 'allPods_hourly_consumption'
np.random.seed(42)
allPods_hourly_consumption = np.random.rand(len(dates_2025), 56) * 2  # Consumi random tra 0 e 2 kWh

df_all = pd.DataFrame(allPods_hourly_consumption, index=dates_2025, columns=[f'POD_{i}' for i in range(1, 57)])

# Separazione dei carichi originali
df_pa = df_all.iloc[:, [i - 1 for i in POD_PA]]
df_illum = df_all.iloc[:, [i - 1 for i in POD_ILLUM]]
df_resid = df_all.iloc[:, [i - 1 for i in POD_RESID]]
df_indus = df_all.iloc[:, [i - 1 for i in POD_INDUS]]

# =============================================================================
# 4. PREPARAZIONE DATI PER LA GAN E TRAINING
# =============================================================================
print(">>> Preparation data residential per la GAN...")
# Rimodelliamo da (8760, 23) a (365*23, 24)
dati_residenziali_array = df_resid.values
num_giorni = len(dates_2025) // 24
num_pod_resid = len(POD_RESID)

profili_giornalieri = dati_residenziali_array.reshape(num_giorni, 24, num_pod_resid)
profili_giornalieri = np.transpose(profili_giornalieri, (0, 2, 1))  # Forma: (365, 23, 24)
profili_giornalieri = profili_giornalieri.reshape(-1, 24)  # Forma: (8395, 24)

# Convertiamo in tensori PyTorch
dataset = torch.tensor(profili_giornalieri, dtype=torch.float32)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Inizializzazione modelli
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
opt_g = optim.Adam(generator.parameters(), lr=LR)
opt_d = optim.Adam(discriminator.parameters(), lr=LR)

print(">>> Avvio addestramento GAN (questo potrebbe richiedere qualche minuto)...")
for epoch in range(EPOCHS):
    for i, real_data in enumerate(dataloader):
        batch_size = real_data.size(0)

        # Etichette reali (1) e fake (0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # --- Train Discriminator ---
        opt_d.zero_grad()
        # Loss sui data reali
        outputs = discriminator(real_data)
        d_loss_real = criterion(outputs, real_labels)

        # Loss sui data falsi (generati)
        z = torch.randn(batch_size, LATENT_DIM)
        fake_data = generator(z)
        outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_d.step()

        # --- Train Generator ---
        opt_g.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM)
        fake_data = generator(z)
        outputs = discriminator(fake_data)

        # Il generatore vuole che il discriminatore classifichi i fake come reali (1)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        opt_g.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoca [{epoch + 1}/{EPOCHS}] | Loss D: {d_loss.item():.4f} | Loss G: {g_loss.item():.4f}")

# =============================================================================
# 5. GENERAZIONE DEI NUOVI PROFILI CON LA GAN
# =============================================================================
num_pod_da_generare = TARGET_FAMIGLIE_CER - num_pod_resid
print(f">>> Generazione di {num_pod_da_generare} nuovi profili residential sintetici...")

nuovi_pod_lista = []
generator.eval()  # Imposta in modalità inferenza

with torch.no_grad():
    for _ in range(num_pod_da_generare):
        # Generiamo 365 giorni (un anno) per un singolo nuovo POD
        z = torch.randn(num_giorni, LATENT_DIM)
        giorni_generati = generator(z).numpy()

        # Appiattiamo i 365 giorni da 24 hours in una singola serie da 8760 hours
        profilo_annuale = giorni_generati.flatten()
        nuovi_pod_lista.append(profilo_annuale)

# Creiamo un DataFrame con i nuovi POD generati
df_resid_gan = pd.DataFrame(np.column_stack(nuovi_pod_lista),
                            index=dates_2025,
                            columns=[f'POD_GAN_{i + 1}' for i in range(num_pod_da_generare)])

# Uniamo i 23 POD originali ai nuovi POD generati
df_resid_completo = pd.concat([df_resid, df_resid_gan], axis=1)

# =============================================================================
# 6. AGGREGAZIONE E SALVATAGGIO
# =============================================================================
print(">>> Computation carichi aggregati e salvataggio file...")

# Carico aggregato residential (Originali + GAN)
carico_residenziali_gan = df_resid_completo.sum(axis=1)

# Carico aggregato TOTALE Scenario CER
# (PA originali + Illum originali + Industriale originali + Residenziali GAN)
carico_totale_cer_gan = (df_pa.sum(axis=1) +
                         df_illum.sum(axis=1) +
                         df_indus.sum(axis=1) +
                         carico_residenziali_gan)

# Saving in Excel (richiede la libreria openpyxl)
df_out_resid = pd.DataFrame({'Timestamp': dates_2025, 'Carico_kWh': carico_residenziali_gan})
df_out_totale = pd.DataFrame({'Timestamp': dates_2025, 'Carico_kWh': carico_totale_cer_gan})

df_out_resid.to_excel('CER_Carico_Residenziali_GAN.xlsx', index=False)
df_out_totale.to_excel('CER_Carico_Orario_Aggregato_GAN.xlsx', index=False)

print(">>> Elaborazione completata successfully! File salvati.")