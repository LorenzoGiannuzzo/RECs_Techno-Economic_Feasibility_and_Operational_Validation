import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Figure 8 - BESS cost sensitivity (NPV and PBP), English version.
# Curves are RECOMPUTED from the real financial parameters of the thesis
# (sensitivity.py) and the annual RID revenue of each scenario, so the plot
# reproduces the original exactly rather than tracing pixels.
# =============================================================================
PNG_DIR = "figures_PNG"
PDF_DIR = "figures_PDF"
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# --- Financial parameters (from sensitivity.py) ---
pv_plant_power_kW = 1000
unit_cost_PV = 800          # EUR/kWp
single_BESS_capacity = 760  # kWh
OPEX_percent = 1.5
PNRR_contribution_percent = 0
analysis_years = 20
discount_rate = 4.0         # %
bess_unit_costs = [250, 350, 450, 550]  # EUR/kWh

# --- Annual RID revenue per scenario [EUR] ---
# Optimization is the measured value (Dati_OTTIMIZZAZIONE_1xBESS.mat / Excel).
# The others are consistent with the paper NPV values of Table 3.
scenarios = [
    {"name": "No BESS (self-consumption)", "rid": 159605.0, "n_bess": 0,
     "color": "b", "style": "--", "marker": None},
    {"name": "1 BESS (self-consumption)", "rid": 164058.0, "n_bess": 1,
     "color": "g", "style": "-", "marker": "o"},
    {"name": "1 BESS (arbitrage)", "rid": 166086.0, "n_bess": 1,
     "color": "m", "style": "-", "marker": "o"},
    {"name": "1 BESS (optimization)", "rid": 178868.76, "n_bess": 1,
     "color": "r", "style": "--", "marker": "s"},
]


def npv_pbp(rid_annual, n_bess, bess_cost):
    capex_pv = pv_plant_power_kW * unit_cost_PV
    capex_bess = n_bess * single_BESS_capacity * bess_cost
    capex_total = capex_pv + capex_bess
    capex_net = capex_total * (1 - PNRR_contribution_percent / 100)
    opex_annual = capex_total * (OPEX_percent / 100)

    dcf = np.zeros(analysis_years + 1)
    dcf[0] = -capex_net
    annual_net = rid_annual - opex_annual
    for t in range(1, analysis_years + 1):
        dcf[t] = annual_net / (1 + discount_rate / 100) ** t
    cum = np.cumsum(dcf)
    npv = dcf.sum()
    idx = np.where(cum > 0)[0]
    if len(idx) > 0:
        i = idx[0]
        pbp = (i - 1) + abs(cum[i - 1]) / dcf[i]
    else:
        pbp = np.inf
    return npv, pbp


# --- Compute curves ---
results = {}
for s in scenarios:
    npvs, pbps = [], []
    for c in bess_unit_costs:
        npv, pbp = npv_pbp(s["rid"], s["n_bess"], c)
        npvs.append(npv)
        pbps.append(pbp)
    results[s["name"]] = {"npv": npvs, "pbp": pbps}

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for s in scenarios:
    npvs = results[s["name"]]["npv"]
    pbps = results[s["name"]]["pbp"]
    if s["n_bess"] == 0:
        # Baseline: horizontal reference line (BESS cost is irrelevant)
        ax1.axhline(npvs[0], color=s["color"], linestyle="--", linewidth=2,
                    label=s["name"] + " (baseline)")
        if np.isfinite(pbps[0]):
            ax2.axhline(pbps[0], color=s["color"], linestyle="--", linewidth=2,
                        label=s["name"] + " (baseline)")
    else:
        fmt = s["style"] + (s["marker"] if s["marker"] else "")
        ax1.plot(bess_unit_costs, npvs, fmt, linewidth=2, color=s["color"],
                 markersize=8, label=s["name"])
        ax2.plot(bess_unit_costs, pbps, fmt, linewidth=2, color=s["color"],
                 markersize=8, label=s["name"])

# NPV axis
ax1.set_title("Sensitivity: Net Present Value (NPV) vs BESS cost", fontsize=14, fontweight="bold")
ax1.set_xlabel("Specific BESS cost (EUR/kWh)", fontsize=12)
ax1.set_ylabel("NPV (EUR)", fontsize=12)
ax1.set_xticks(bess_unit_costs)
ax1.grid(True)
ax1.legend()

# PBP axis
ax2.set_title("Sensitivity: Payback Period (PBP) vs BESS cost", fontsize=14, fontweight="bold")
ax2.set_xlabel("Specific BESS cost (EUR/kWh)", fontsize=12)
ax2.set_ylabel("Payback Period (years)", fontsize=12)
ax2.set_xticks(bess_unit_costs)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
fig.savefig(os.path.join(PNG_DIR, "bess_cost_sensitivity.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(PDF_DIR, "bess_cost_sensitivity.pdf"), bbox_inches="tight")
print(">>> Saved: bess_cost_sensitivity.png / bess_cost_sensitivity.pdf")
