"""Step 1 - design-stage analysis: dispatch scenarios, financial analysis with degradation, BESS cost sensitivity,
and benefit allocation (Sections 2.1-2.3, results in Sections 3.2, 4.1, and 4.2)."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import config as C
from .data import NON_BUSINESS
from .models import econ, milp, rule


def run(inp, paths, log=print):
    PV, PZ, TIP, DEM, CATS = inp.pv, inp.pz, inp.tip, inp.dem, inp.cats
    out = paths.results_dir
    R = {}
    R["inputs"] = dict(pv_MWh=PV.sum() / 1e3, demand_MWh=DEM.sum() / 1e3,
                       cat_MWh={k: v.sum() / 1e3 for k, v in CATS.items()},
                       price_mean=PZ.mean(), tip_cap_share=float((PZ <= 140).mean() * 100),
                       tip_var_share=float(((PZ > 140) & (PZ < 180)).mean() * 100),
                       tip_fixed_share=float((PZ >= 180).mean() * 100), tip_mean=TIP.mean())
    np.savez(out / "loads_chrono.npz", ts=inp.ts_meas.astype("int64"), **inp.cats_chrono)

    #Lorenzo Giannuzzo: monthly synthesis of the PV generation (Table 3)
    pvdf = pd.DataFrame({"pv": PV}, index=inp.ts_sim)
    tab = []
    for mth in range(1, 13):
        s = pvdf[pvdf.index.month == mth].pv
        prof = s.groupby(s.index.hour).mean()
        tab.append(dict(month=mth, peak=float(prof.max()),
                        mean_9_17=float(s[(s.index.hour >= 9) & (s.index.hour < 17)].mean()),
                        daily=float(s.sum() / s.index.day.nunique()), monthly=float(s.sum() / 1e3)))
    R["table1"] = dict(rows=tab, year_daily=float(PV.sum() / 365), year_MWh=float(PV.sum() / 1e3),
                       specific_yield=float(PV.sum() / C.P_DC),
                       share_apr_sep=float(100 * pvdf[(pvdf.index.month >= 4) & (pvdf.index.month <= 9)].pv.sum()
                                           / PV.sum()))

    def vsc(sh):
        w = np.divide(sh, DEM, out=np.zeros_like(sh), where=DEM > 0)
        return {k: 100 * (w * v).sum() / v.sum() for k, v in CATS.items()}

    #Lorenzo Giannuzzo: design-stage scenarios on the first year (Tables 4 and 5)
    log(">>> Dispatch scenarios")
    SC = {"No-BESS": econ(PV.copy(), DEM, PZ, TIP) | {"pc": np.zeros(C.HOURS), "pd": np.zeros(C.HOURS)}}
    for name, mode in [("Self-consumption", "sc"), ("Arbitrage", "arb")]:
        r = rule(PV, DEM, PZ, mode); SC[name] = econ(PV - r["pc"] + r["pd"], DEM, PZ, TIP) | r
    mm = milp(PV, DEM, PZ, TIP)
    SC["MILP"] = econ(PV - mm["pc"] + mm["pd"], DEM, PZ, TIP) | mm
    R["milp_comp"] = dict(time_s=mm["time"], nvar=mm["nvar"], ncon=mm["ncon"], nbin=mm["nbin"], status=mm["status"])
    es_gap = np.abs(mm["es"] - np.minimum(PV - mm["pc"] + mm["pd"], DEM)).max()
    simult = int(((mm["pc"] > 1e-6) & (mm["pd"] > 1e-6)).sum())
    R["milp_checks"] = dict(max_dev_shared_vs_min_kWh=float(es_gap), simultaneous_hours=simult)
    base = SC["No-BESS"]
    R["scenarios"] = {}
    for k, v in SC.items():
        R["scenarios"][k] = dict(shared_MWh=v["shared_MWh"], inj_MWh=v["inj_MWh"], inc=v["inc"], arera=v["arera"],
                                 rid=v["rid"], gross=v["gross"], rho=v["rho"],
                                 d_shared=100 * (v["shared_MWh"] / base["shared_MWh"] - 1),
                                 d_gross=100 * (v["gross"] / base["gross"] - 1),
                                 cycles=float(v["pd"].sum() / C.ETA / C.E_NOM) if v["pd"].sum() > 0 else 0.0,
                                 vsc=vsc(v["sh"]))
    inj_m = SC["MILP"]["inj"]; pvh = PV > 0
    R["milp_demand_limited"] = dict(share_pv_hours_dem_ge_inj=float(100 * ((DEM >= inj_m) & pvh).sum() / pvh.sum()))
    np.savez(out / "milp_year.npz", pv=PV, pz=PZ, dem=DEM, pc=mm["pc"], pd=mm["pd"], inj=inj_m, sh=SC["MILP"]["sh"],
             **{f"cat_{i}": v for i, v in enumerate(CATS.values())})
    log("    shared energy [MWh] / gross revenue [€]: "
        + ", ".join(f"{k} {v['shared_MWh']:.1f} / {v['gross']:.0f}" for k, v in SC.items()))

    #Lorenzo Giannuzzo: discounted cash flow over the incentive period, with PV degradation and a linear capacity fade
    # of the battery driven by the energy throughput; the dispatch is recomputed for every year (Eqs. 18-22, Table 6)
    log(">>> Discounted cash flow")

    def yearly_rid(name, degrade=True):
        soh, res, cyc_cum = 1.0, [], 0.0
        for y in range(1, C.N_YEARS + 1):
            pv_y = PV * ((1 - C.PV_DEG) ** (y - 1) if degrade else 1.0)
            e_y = C.E_NOM * (soh if degrade else 1.0)
            if name == "No-BESS":
                r = econ(pv_y, DEM, PZ, TIP); cyc = 0.0
            elif name == "MILP":
                if not degrade and y > 1:
                    res.append(res[-1]); continue
                d = milp(pv_y, DEM, PZ, TIP, e_nom=e_y); r = econ(pv_y - d["pc"] + d["pd"], DEM, PZ, TIP)
                cyc = d["pd"].sum() / C.ETA / C.E_NOM
            else:
                if not degrade and y > 1:
                    res.append(res[-1]); continue
                d = rule(pv_y, DEM, PZ, "sc" if name == "Self-consumption" else "arb", e_nom=e_y)
                r = econ(pv_y - d["pc"] + d["pd"], DEM, PZ, TIP); cyc = d["pd"].sum() / C.ETA / C.E_NOM
            res.append(dict(year=y, soh=soh, rid=r["rid"], shared_MWh=r["shared_MWh"], gross=r["gross"], cycles=cyc))
            cyc_cum += cyc
            if degrade:
                soh = 1.0 - (1.0 - C.SOH_EOL) * cyc_cum / C.CYCLE_LIFE
        return res

    def dcf(rids, n_bess, c_bess):
        capex = C.P_DC * C.C_PV + n_bess * C.E_NOM * c_bess; opex = C.OPEX_PCT / 100 * capex
        cf = np.array([r - opex for r in rids])
        disc = cf / (1 + C.RATE) ** np.arange(1, C.N_YEARS + 1)
        npv = disc.sum() - capex
        cum = np.r_[-capex, -capex + np.cumsum(disc)]
        i = np.where(cum > 0)[0]
        pbp = (i[0] - 1) + abs(cum[i[0] - 1]) / disc[i[0] - 1] if len(i) else np.inf
        irr = brentq(lambda r: (cf / (1 + r) ** np.arange(1, C.N_YEARS + 1)).sum() - capex, -0.5, 1.0)
        return dict(capex=capex, opex=opex, npv=npv, pbp=pbp, irr=100 * irr, cum=cum.tolist())

    R["dcf"], R["dcf_nodeg"], R["sens"], YR = {}, {}, {}, {}
    for name in ["No-BESS", "Self-consumption", "Arbitrage", "MILP"]:
        nb = 0 if name == "No-BESS" else 1
        YR[name] = yearly_rid(name, True)
        rids = [y["rid"] for y in YR[name]]
        R["dcf"][name] = dcf(rids, nb, C.C_BESS) | dict(
            rid_y1=rids[0], rid_y20=rids[-1], soh_y20=YR[name][-1]["soh"],
            soh_end=1.0 - (1.0 - C.SOH_EOL) * sum(y["cycles"] for y in YR[name]) / C.CYCLE_LIFE,
            cycles_total=sum(y["cycles"] for y in YR[name]))
        R["dcf_nodeg"][name] = dcf([SC[name]["rid"]] * C.N_YEARS, nb, C.C_BESS)
        R["sens"][name] = {c: {k: v for k, v in dcf(rids, nb, c).items() if k != "cum"} for c in C.SENS_COSTS}
        d_ = R["dcf"][name]
        log(f"    {name}: NPV {d_['npv']:.0f} €, IRR {d_['irr']:.1f}%, PBP {d_['pbp']:.2f} years")
    R["yearly"] = YR
    #Lorenzo Giannuzzo: break-even BESS cost of MILP against the no-BESS baseline (the NPV is linear in the unit cost)
    npv0 = R["dcf"]["No-BESS"]["npv"]
    a, b = R["sens"]["MILP"][C.SENS_COSTS[0]]["npv"], R["sens"]["MILP"][C.SENS_COSTS[1]]["npv"]
    slope = (b - a) / (C.SENS_COSTS[1] - C.SENS_COSTS[0])
    R["breakeven_cost"] = C.SENS_COSTS[0] + (npv0 - a) / slope if slope != 0 else None
    R["npv_slope_per_eur_kWh"] = slope
    opex_bess = R["dcf"]["MILP"]["opex"]
    R["opex_cov"] = dict(opex_milp=opex_bess, rid_min_year=min(y["rid"] for y in YR["MILP"]),
                         price_threshold=opex_bess / (SC["MILP"]["inj_MWh"]))

    #Lorenzo Giannuzzo: three-phase CACER allocation of the MILP revenue, with the surplus above the 55% threshold
    # reserved for the non-business members (Eqs. 23-31, Table 7)
    m1 = SC["MILP"]
    rho = m1["rho"]; theta = max(0.0, 1.0 - C.RHO_STAR / rho)
    entity = m1["rid"] + C.ALPHA * m1["inc"]
    dist = (1 - C.ALPHA) * m1["inc"] + m1["arera"]
    sur = theta * (1 - C.ALPHA) * m1["inc"]; ordq = dist - sur
    Ecat = {k: v.sum() for k, v in CATS.items()}
    Eall = sum(Ecat.values()); Enb = sum(Ecat[k] for k in NON_BUSINESS)
    alloc = {k: dict(ordinary=ordq * Ecat[k] / Eall, surplus=(sur * Ecat[k] / Enb if k in NON_BUSINESS else 0.0))
             for k in CATS}
    for k in alloc:
        alloc[k]["total"] = alloc[k]["ordinary"] + alloc[k]["surplus"]
    R["alloc"] = dict(theta=theta, rho=rho, entity=entity, dist=dist, surplus=sur, ordinary=ordq, per_cat=alloc,
                      gross=m1["gross"], per_household=alloc["Residential"]["total"] / C.N_HOUSEHOLDS,
                      per_household_surplus=alloc["Residential"]["surplus"] / C.N_HOUSEHOLDS,
                      business_share_of_incentive=100 * alloc["Industrial"]["total"] / dist)
    json.dump(R, open(out / "results_design.json", "w"), indent=1, default=float)
    log(">>> Step 1 (design) done")
    return R
