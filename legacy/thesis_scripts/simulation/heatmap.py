import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn library

# =============================================================================
# 0. OUTPUT HELPER (saves every figure as PNG and PDF in two separate folders)
# =============================================================================
PNG_DIR = "figures_PNG"
PDF_DIR = "figures_PDF"
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)


def save_figure(fig, name, dpi=300):
    """Save a figure as PNG and PDF into the two dedicated folders."""
    fig.savefig(os.path.join(PNG_DIR, name + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(PDF_DIR, name + ".pdf"), bbox_inches="tight")
    print(f">>> Saved: {name}.png / {name}.pdf")


# =============================================================================
# 1. SETTINGS AND DATA LOADING
# =============================================================================
print(">>> Starting annual heatmap creation in Python with Seaborn...")

aggregated_file = "CER_Carico_Orario_Aggregato_k.xlsx"
try:
    aggregated_df = pd.read_excel(aggregated_file)
    aggregated_df["Timestamp"] = pd.to_datetime(aggregated_df["Timestamp"])
except FileNotFoundError:
    print(f"ERROR: File not found: {aggregated_file}. Make sure the script sits next to the Excel file.")
    exit()

total_load = aggregated_df["Carico_kWh"].values
time_vector = aggregated_df["Timestamp"]

print(">>> Data loaded successfully.")

# =============================================================================
# 2. DATA PREPARATION FOR THE TWO HEATMAPS
# =============================================================================
# --- HEATMAP 1: RAW DATA (ORIGINAL DATA) ---
# Reshape the vector into a [365 days x 24 hours] matrix for a vertical layout
raw_heatmap_data = total_load.reshape(365, 24)

# --- HEATMAP 2: CLEANED DATA ---
# 1. Replace values < 10 with NaN (Not a Number)
load_with_nan = np.where(total_load < 10, np.nan, total_load)

# 2. Reshape into a 365x24 matrix
nan_heatmap_data = load_with_nan.reshape(365, 24)

# 3. Use Pandas for an easy linear interpolation along the columns (days)
interpolation_df = pd.DataFrame(nan_heatmap_data)
interpolated_df = interpolation_df.interpolate(method="linear", axis=0, limit_direction="both")
interpolated_heatmap_data = interpolated_df.to_numpy()

print(">>> Data prepared for both heatmaps (original and cleaned).")

# --- Prepare the labels for the month axis (now on the Y axis) ---
first_day_of_year = time_vector.iloc[0].normalize()
y_tick_positions = []
y_tick_labels = []
for month in range(1, 13):
    try:
        first_day_of_month_ts = time_vector[time_vector.dt.month == month].iloc[0]
        day_of_year = (first_day_of_month_ts.normalize() - first_day_of_year).days
        if day_of_year not in y_tick_positions:
            y_tick_positions.append(day_of_year)
            y_tick_labels.append(first_day_of_month_ts.strftime("%b-%y"))
    except IndexError:
        # If a month is not present in the data, skip it
        continue

# For Seaborn we use DataFrames. The (365, 24) shape is already correct for a vertical layout.
raw_df = pd.DataFrame(raw_heatmap_data, index=range(365), columns=range(24))
interpolated_df = pd.DataFrame(interpolated_heatmap_data, index=range(365), columns=range(24))

# Find the global maximum value for a consistent color scale
vmax = np.nanmax(interpolated_df.values)

# =============================================================================
# 3. HEATMAP FIGURES (SEPARATE AND VERTICAL)
# =============================================================================

# --- Figure 1: Heatmap with raw data (Original Data) ---
fig1, ax1 = plt.subplots(figsize=(10, 16))  # Size tuned for the vertical layout
sns.heatmap(raw_df, ax=ax1, cmap="Spectral_r",
            cbar_kws={"label": "Electrical energy [kWh]"}, vmin=0, vmax=vmax)
ax1.set_title("Heatmap - Original Data", fontsize=16, fontweight="bold")
ax1.set_xlabel("Hour of the day")
ax1.set_ylabel("Month of the year")
ax1.set_xticks(range(0, 25, 4))
ax1.set_xticklabels(range(0, 25, 4))
ax1.set_yticks(y_tick_positions)
ax1.set_yticklabels(y_tick_labels, rotation=0)
fig1.tight_layout()
save_figure(fig1, "heatmap_original_data")

# --- Figure 2: Heatmap with interpolated data (Cleaned Data) ---
fig2, ax2 = plt.subplots(figsize=(10, 16))
sns.heatmap(interpolated_df, ax=ax2, cmap="Spectral_r",
            cbar_kws={"label": "Electrical energy [kWh]"}, vmin=0, vmax=vmax)
ax2.set_title("Heatmap - Cleaned Data", fontsize=16, fontweight="bold")
ax2.set_xlabel("Hour of the day")
ax2.set_ylabel("Month of the year")
ax2.set_xticks(range(0, 25, 4))
ax2.set_xticklabels(range(0, 25, 4))
ax2.set_yticks(y_tick_positions)
ax2.set_yticklabels(y_tick_labels, rotation=0)
fig2.tight_layout()
save_figure(fig2, "heatmap_cleaned_data")

print(">>> Heatmaps created successfully.")

# --- Figure 3: Aggregated REC Consumption Heatmap ---
fig3, ax3 = plt.subplots(figsize=(10, 16))
sns.heatmap(interpolated_df, ax=ax3, cmap="Spectral_r",
            cbar_kws={"label": "Electrical energy [kWh]"}, vmin=0, vmax=vmax)

# 1. TITLE: enlarged to 28 and spaced out (pad=25)
ax3.set_title("Aggregated REC consumption", fontsize=28, fontweight="bold", pad=25)

# 2. AXIS LABELS: enlarged to 24 and spaced out (labelpad=20)
ax3.set_xlabel("Hour of the day", fontsize=24, labelpad=20)
ax3.set_ylabel("Month of the year", fontsize=24, labelpad=20)

# --- 3. MONTHS ON THE Y AXIS ---
days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_y_positions = []
cumulative_days = 0
for days in days_in_months:
    month_y_positions.append(cumulative_days + (days / 2))
    cumulative_days += days

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ax3.set_yticks(month_y_positions)
ax3.set_yticklabels(month_labels, rotation=0, fontsize=18)

# X axis formatting (hours enlarged to 18)
ax3.tick_params(axis="x", labelsize=18)

# 4. COLORBAR
cbar = ax3.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
cbar.set_label("Electrical energy [kWh]", size=24, labelpad=20)

fig3.tight_layout()
save_figure(fig3, "heatmap_aggregated_consumption")

print(">>> All heatmaps saved as PNG and PDF.")
