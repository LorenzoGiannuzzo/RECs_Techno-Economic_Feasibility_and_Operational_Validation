"""Step 5 - figures of the paper (PNG and vector PDF at 500 dpi), drawn from the outputs of steps 1-4."""
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                     "font.size": 11,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#d0d0d0",
                     "grid.linewidth": 0.6, "axes.edgecolor": "black", "legend.edgecolor": "black",
                     "legend.fancybox": False, "legend.framealpha": 1.0, "savefig.dpi": 500, "pdf.fonttype": 42,
                     "axes.formatter.use_locale": False})


def TSP(a):
    a = np.asarray(a)
    return pd.to_datetime(a, unit="us" if a.max() < 1e17 else "ns")


def run(paths, log=print):
    OUT, FIG = str(paths.results_dir), str(paths.figures_dir)
    RD = json.load(open(f"{OUT}/results_design.json")); RO = json.load(open(f"{OUT}/results_ops.json"))

    def save(fig, name):
        fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log(f"    {name}")

    PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
    COL4 = {"No-BESS": "#1f4e9c", "Self-consumption": "#2e8b3a", "Arbitrage": "#b0399c", "MILP": "#d62728"}
    LAB4 = {"No-BESS": "No-BESS Baseline", "Self-consumption": "BESS Self-Consumption", "Arbitrage": "BESS Arbitrage",
            "MILP": "BESS MILP"}
    HOLI = pd.to_datetime(["2025-01-01", "2025-01-06", "2025-04-20", "2025-04-21", "2025-04-25", "2025-05-01", "2025-06-02",
                           "2025-08-15", "2025-11-01", "2025-12-08", "2025-12-25", "2025-12-26",
                           "2024-11-01", "2024-12-08", "2024-12-25", "2024-12-26"]).date



    #Lorenzo Giannuzzo: Fig. 2 - seasonal hourly profiles by day of the week (measured, chronological calendar)
    L = np.load(f"{OUT}/loads_chrono.npz")
    ts = TSP(L["ts"])
    cats = [("Public administration", "Public Administration"), ("Street lighting", "Street Lighting"),
            ("Residential", "Residential Users"), ("Industrial", "Industrial User")]
    seasons = [("Winter (December–February)", [12, 1, 2]),
               ("Intermediate Seasons (March–May, September–November)", [3, 4, 5, 9, 10, 11]),
               ("Summer (June–August)", [6, 7, 8])]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dcol = ["#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30", "#4dbeee", "#a2142f"]
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
    for r, (key, title) in enumerate(cats):
        df = pd.DataFrame({"v": L[key]}, index=ts)
        profs = []
        for c, (sname, months) in enumerate(seasons):
            s = df[df.index.month.isin(months)]
            pr = s.groupby([s.index.dayofweek, s.index.hour]).v.mean().unstack(0)
            profs.append(pr)
        ymax = max(p.values.max() for p in profs) * 1.05
        for c, pr in enumerate(profs):
            ax = axes[r, c]
            for dw in range(7):
                ax.plot(pr.index, pr[dw], "-o", ms=2.5, lw=1.2, color=dcol[dw], label=days[dw])
            ax.set_ylim(0, ymax); ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 2))
            ax.yaxis.set_major_formatter(PLAIN)
            ax.set_title(f"{title} – {seasons[c][0].split(' (')[0]}", fontsize=11)
            if c == 0:
                ax.set_ylabel("Mean Hourly Consumption [kWh]")
            if r == 3:
                ax.set_xlabel("Hour [h]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.005), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "Fig02_Seasonal_Load_Profiles")

    #Lorenzo Giannuzzo: Fig. 3 - annual heatmaps of the MILP scenario
    Y = np.load(f"{OUT}/milp_year.npz")
    panels = [("PV Generation [kW]", Y["pv"], "Power [kW]"), ("Zonal Market Price [€/MWh]", Y["pz"], "Price [€/MWh]"),
              ("BESS Charging Power [kW]", Y["pc"], "Power [kW]"), ("BESS Discharging Power [kW]", Y["pd"], "Power [kW]"),
              ("Grid Injection [kW]", Y["inj"], "Power [kW]")]
    fig = plt.figure(figsize=(13, 10.5))
    pos = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)]
    gs = fig.add_gridspec(3, 4)
    for (title, data, cl), (rr, cc) in zip(panels, pos):
        ax = fig.add_subplot(gs[rr, cc:cc + 2])
        im = ax.imshow(np.asarray(data).reshape(365, 24).T, aspect="auto", cmap="plasma", interpolation="nearest",
                       origin="upper", extent=[0, 365, 23.5, -0.5])
        ax.set_title(title, fontsize=11); ax.grid(False)
        ax.set_ylabel("Hour [h]"); ax.set_xlabel("Day [d]")
        cb = fig.colorbar(im, ax=ax); cb.set_label(cl); cb.ax.yaxis.set_major_formatter(PLAIN)
    fig.tight_layout()
    save(fig, "Fig03_MILP_Operational_Heatmaps")

    #Lorenzo Giannuzzo: Fig. 4 - typical days of the MILP scenario, seasons x day type
    T5 = pd.date_range("2025-01-01", periods=8760, freq="h")
    df = pd.DataFrame({"sh": Y["sh"], "dem": Y["dem"], "inj": Y["inj"]}, index=T5)
    E = 760.0; eta = 0.95 ** 0.5
    soc = np.zeros(8760); s = 0.5 * E
    for t in range(8760):
        s += Y["pc"][t] * eta - Y["pd"][t] / eta; soc[t] = 100 * s / E
    df["soc"] = soc
    isw = (df.index.dayofweek >= 5) | pd.Series(df.index.date, index=df.index).isin(HOLI).values
    smap = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer",
            8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df.index.month.map(smap)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharey=True)
    for c, sn in enumerate(["Winter", "Spring", "Summer", "Autumn"]):
        for r, (lbl, mask) in enumerate([("Weekday", ~isw), ("Weekend/Holiday", isw)]):
            ax = axes[r, c]; sub = df[(df.season == sn) & mask]
            p = sub.groupby(sub.index.hour).mean(numeric_only=True)
            ax.fill_between(p.index, p.sh, color="#5cc85c", alpha=0.85, label="Shared Energy")
            ax.plot(p.index, p.dem, color="#d62728", lw=2, label="Aggregated REC Demand")
            ax.plot(p.index, p.inj, ":", color="black", lw=1.6, label="Total Injected Energy")
            ax2 = ax.twinx(); ax2.plot(p.index, p.soc, "--", color="#b0399c", lw=1.6, label="Average Battery SOC")
            ax2.set_ylim(0, 100); ax2.grid(False)
            if c == 3:
                ax2.set_ylabel("State Of Charge [%]", color="#b0399c")
            else:
                ax2.set_yticklabels([])
            ax2.tick_params(axis="y", colors="#b0399c")
            ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 4)); ax.yaxis.set_major_formatter(PLAIN)
            ax.set_title(f"{sn} – {lbl}", fontsize=11)
            if c == 0:
                ax.set_ylabel("Energy [kWh]")
            if r == 1:
                ax.set_xlabel("Hour [h]")
    h1, l1 = axes[0, 0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "Fig04_Typical_Daily_Profiles")

    #Lorenzo Giannuzzo: Fig. 5 - discounted cumulative cash flow (with degradation)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for k in ["No-BESS", "Self-consumption", "Arbitrage", "MILP"]:
        cum = np.array(RD["dcf"][k]["cum"]) / 1e3
        ax.plot(range(21), cum, "-o" if k != "MILP" else "--s", color=COL4[k], ms=4, lw=1.8, label=LAB4[k])
        pbp = RD["dcf"][k]["pbp"]
        ax.axvline(pbp, color=COL4[k], ls="--", lw=1); ax.plot(pbp, 0, "o", ms=7, mfc=COL4[k], mec="black")
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xlabel("Year [-]"); ax.set_ylabel("Discounted Cumulative Cash Flow [k€]")
    ax.set_xticks(range(0, 21, 2)); ax.set_xlim(-0.5, 20.5); ax.yaxis.set_major_formatter(PLAIN)
    ax.legend(loc="upper center", ncol=4, frameon=True)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    save(fig, "Fig05_Discounted_Cash_Flow")

    #Lorenzo Giannuzzo: Fig. 6 - BESS cost sensitivity
    costs = [150, 250, 350, 450, 550]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    for k in ["Self-consumption", "Arbitrage", "MILP"]:
        npv = [RD["sens"][k][str(c)]["npv"] / 1e3 for c in costs]; pbp = [RD["sens"][k][str(c)]["pbp"] for c in costs]
        a1.plot(costs, npv, "-o" if k != "MILP" else "--s", color=COL4[k], lw=1.8, ms=6, label=LAB4[k])
        a2.plot(costs, pbp, "-o" if k != "MILP" else "--s", color=COL4[k], lw=1.8, ms=6, label=LAB4[k])
    a1.axhline(RD["dcf"]["No-BESS"]["npv"] / 1e3, color=COL4["No-BESS"], ls="--", lw=1.8, label=LAB4["No-BESS"])
    a2.axhline(RD["dcf"]["No-BESS"]["pbp"], color=COL4["No-BESS"], ls="--", lw=1.8, label=LAB4["No-BESS"])
    for a, yl, t in [(a1, "Net Present Value [k€]", "Net Present Value"), (a2, "Discounted Payback Period [Years]", "Payback Period")]:
        a.set_xlabel("BESS Unit Cost [€/kWh]"); a.set_ylabel(yl); a.set_xticks(costs); a.yaxis.set_major_formatter(PLAIN)
        a.set_title(t)
        a.legend(loc="upper center", ncol=2, frameon=True)
        lo, hi = a.get_ylim(); a.set_ylim(lo, hi + (hi - lo) * 0.28)
    fig.tight_layout()
    save(fig, "Fig06_BESS_Cost_Sensitivity")

    #Lorenzo Giannuzzo: Fig. 8 - out-of-sample forecast window
    F = np.load(f"{OUT}/forecast.npz")
    tf = TSP(F["ts"])
    w = (tf >= "2025-03-03") & (tf < "2025-03-24")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(tf[w], F["y"][w], "-", color="#0072bd", lw=1.5, label="Observed")
    ax.plot(tf[w], F["pred"][w], "--", color="#d95319", lw=1.5, label="Predicted (Rolling-Origin)")
    ax.set_ylabel("Residential Hourly Consumption [kWh]"); ax.set_xlabel("Date [-]")
    ax.yaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m")); ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.legend(loc="upper center", ncol=2, frameon=True)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * 0.15)
    save(fig, "Fig08_Rolling_Origin_Forecast")

    #Lorenzo Giannuzzo: Fig. 9 - SHAP beeswarm (out-of-sample, kWh)
    import shap
    FEAT = ["Hour (Sine)", "Hour (Cosine)", "Day Of The Week (Sine)", "Day Of The Week (Cosine)", "Non-Working Day",
            "Load 24 h Before", "Load 168 h Before", "Load 48 h Before", "Temperature Change Over 24 h",
            "Outdoor Temperature"]
    plt.figure()
    shap.summary_plot(F["shap"], F["shapX"], feature_names=FEAT, show=False, plot_size=(9, 6.2))
    fig = plt.gcf(); ax = fig.axes[0]; ax.set_xlabel("SHAP Value [kWh]"); ax.xaxis.set_major_formatter(PLAIN)
    for a_ in fig.axes[1:]:
        a_.set_ylabel("Feature Value")
    save(fig, "Fig09_SHAP_Summary")

    #Lorenzo Giannuzzo: Fig. 10 - critical week of August (plan B, all users forecast)
    B = np.load(f"{OUT}/plan_B.npz")
    t11 = pd.date_range("2025-01-01", periods=len(B["pf_sh"]), freq="h")
    d = pd.DataFrame({"pf": B["pf_sh"], "ol": B["real_sh"]}, index=t11)
    aug = d[d.index.month == 8]
    worst = (aug.pf - aug.ol).groupby(aug.index.date).sum().idxmax()
    start = pd.Timestamp(worst) - pd.Timedelta(days=3); end = start + pd.Timedelta(days=7)
    wk = d[(d.index >= start) & (d.index < end)]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(wk.index, wk.pf, "--", color="#2e8b3a", lw=2, label="Perfect-Foresight MILP Benchmark")
    ax.plot(wk.index, wk.ol, "-", color="#d62728", lw=1.3, label="Open-Loop Forecast-Driven Operation")
    ax.set_ylabel("Shared Energy [kWh]"); ax.set_xlabel("Date [-]"); ax.yaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m")); ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.legend(loc="upper center", ncol=2, frameon=True)
    lo, hi = ax.get_ylim(); ax.set_ylim(0, hi * 1.18)
    save(fig, "Fig10_Critical_Week")
    wk_loss = ((wk.pf - wk.ol).clip(lower=0)).sum()
    json.dump(dict(week_start=str(start.date()), week_end=str((end - pd.Timedelta(days=1)).date()), worst_day=str(worst),
                   week_shared_pf=float(wk.pf.sum()), week_shared_ol=float(wk.ol.sum()),
                   week_gap_pct=float(100 * (wk.pf.sum() - wk.ol.sum()) / wk.pf.sum())),
              open(f"{OUT}/critical_week.json", "w"), indent=1)


    #Lorenzo Giannuzzo: Fig. 7 - demand archetypes identified by hierarchical clustering
    K = np.load(f"{OUT}/clusters.npz"); lab = K["labels"]
    hours = np.arange(24)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for i, ax in enumerate(axes):
        X = K["profiles"][lab == i]; c = X.mean(0); sd = X.std(0, ddof=1)
        rmse = np.sqrt(np.mean(np.mean((X - c) ** 2, axis=1)))
        for row in X:
            ax.plot(hours, row, color="grey", alpha=0.25, lw=0.8, zorder=1)
        ax.fill_between(hours, c - sd, c + sd, color="#d62728", alpha=0.2, lw=0, zorder=2)
        ax.plot(hours, c, "--", color="#d62728", lw=2.2, zorder=3)
        ax.set_title(f"Cluster {i + 1} ({len(X)} Days)")
        ax.text(0.04, 0.05, f"RMSE: {rmse:.3f}", transform=ax.transAxes, va="bottom",
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black"))
        ax.set_xlabel("Hour Of The Day [h]"); ax.set_xticks(range(0, 24, 3)); ax.set_xlim(0, 23)
        ax.set_ylim(0, 1.25)
    axes[0].set_ylabel("Normalized Energy [-]")
    handles = [Line2D([], [], color="grey", lw=1.2, alpha=0.6, label="Daily Profiles"),
               Line2D([], [], color="#d62728", ls="--", lw=2.2, label="Cluster Centroid"),
               Patch(color="#d62728", alpha=0.2, label="Centroid ± 1 Standard Deviation")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, "Fig07_Demand_Archetypes")


    log(">>> Step 5 (figures) done")
