import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 0. OUTPUT HELPER (PNG + PDF into two separate folders)
# =============================================================================
PNG_DIR = "figures_PNG"
PDF_DIR = "figures_PDF"
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)


def save_figure(fig, name, dpi=300):
    fig.savefig(os.path.join(PNG_DIR, name + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(PDF_DIR, name + ".pdf"), bbox_inches="tight")
    print(f">>> Saved: {name}.png / {name}.pdf")


# --- 1. Configuration ---
# Original PVsyst hourly export (used in the thesis). Kept here for reference.
pvsyst_file = "local_data/CaseStudy_Project_VC0_HourlyRes_0.CSV"
# Portable fallback: PVGIS hourly time series shipped with the project.
pvgis_file = "pvgis_data.csv"

# System nominal power [kWp]: the case-study plant is 1 MWp.
PLANT_KWP = 1000.0

sns.set_theme(style="whitegrid", palette="viridis")

# --- 2. Data reading and preparation ---
df = None
if os.path.exists(pvsyst_file):
    # Original PVsyst path (semicolon separated, comma decimal, 13 header rows)
    df = pd.read_csv(pvsyst_file, skiprows=13, header=None,
                     names=["datetime", "E_Grid_kW"], sep=";", decimal=",")
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%y %H:%M")
    df.set_index("datetime", inplace=True)
    df["E_Grid_kW"] = pd.to_numeric(df["E_Grid_kW"])
    print("Data loaded from the PVsyst export.")
else:
    # Portable PVGIS fallback: column P is the AC power for a 1 kWp system [W].
    raw = pd.read_csv(pvgis_file, skiprows=10, sep=",")
    raw = raw[raw["time"].astype(str).str.match(r"^\d{8}:")].copy()
    raw["datetime"] = pd.to_datetime(raw["time"], format="%Y%m%d:%H%M")
    raw["E_Grid_kW"] = pd.to_numeric(raw["P"]) / 1000.0 * PLANT_KWP  # W(1 kWp) -> kW at plant scale
    df = raw[["datetime", "E_Grid_kW"]].set_index("datetime")
    print("Data loaded from the PVGIS time series (scaled to the plant nominal power).")

# --- 3. High-quality seasonal figure ---
print("Creating the seasonal average hourly production profiles...")

df["month"] = df.index.month
df["hour"] = df.index.hour

fig1, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=True)
fig1.suptitle("Seasonal average hourly production profiles", fontsize=20, weight="bold")

seasons = {
    "Winter (DJF)": [12, 1, 2],
    "Spring (MAM)": [3, 4, 5],
    "Summer (JJA)": [6, 7, 8],
    "Autumn (SON)": [9, 10, 11],
}


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


for ax, (season_name, months) in zip(axes.flatten(), seasons.items()):
    season_df = df[df["month"].isin(months)]

    # Hourly statistics: mean, 25th and 75th percentile
    profile_stats = season_df.groupby("hour")["E_Grid_kW"].agg(["mean", q25, q75])

    # Mean line
    ax.plot(profile_stats.index, profile_stats["mean"],
            marker="o", markersize=4, linestyle="-", linewidth=2.5,
            label="Mean hourly production")

    # Shaded variability band (interquartile range)
    ax.fill_between(profile_stats.index, profile_stats["q25"], profile_stats["q75"],
                    alpha=0.2, label="Interquartile range (25th-75th percentile)")

    # Annotate the mean power peak
    peak_power = profile_stats["mean"].max()
    peak_hour = profile_stats["mean"].idxmax()
    ax.annotate(f"Peak: {peak_power:.1f} kW",
                xy=(peak_hour, peak_power),
                xytext=(peak_hour, peak_power + 0.05 * peak_power + 5),
                ha="center",
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

    ax.set_title(season_name, fontsize=16, weight="bold")
    ax.set_xlabel("Hour of the day", fontsize=12)
    ax.set_ylabel("Power [kW]", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_figure(fig1, "pv_seasonal_profiles")
plt.close(fig1)

# --- 4. Carpet plot (vertical orientation) ---
print("Creating the annual hourly production carpet plot...")

production_pivot = df.pivot_table(values="E_Grid_kW", index="month", columns="hour", aggfunc="mean")

fig2, ax2 = plt.subplots(figsize=(10, 16))
sns.heatmap(production_pivot, cmap="viridis", ax=ax2,
            cbar_kws={"label": "Mean power [kW]"})

ax2.set_title("Monthly average hourly production", fontsize=30, weight="bold", pad=20)
ax2.set_xlabel("Hour of the day", fontsize=24, labelpad=15)
ax2.set_ylabel("Month", fontsize=24, labelpad=15)

plt.yticks(ticks=[i + 0.5 for i in range(12)],
           labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           rotation=0, fontsize=20)
plt.xticks(fontsize=20, rotation=0)

cbar = ax2.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Mean power [kW]", size=24, labelpad=15)

plt.tight_layout()
save_figure(fig2, "pv_carpet_plot_annual")
plt.close(fig2)

print("\nScript completed. Figures saved as PNG and PDF.")
