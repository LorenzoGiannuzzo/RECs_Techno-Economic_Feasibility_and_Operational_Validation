"""Dispatch models (Section 2.2) and hourly economic evaluation of the REC (Section 2.1)."""
import time

import numpy as np
import pulp

from . import config as C


def milp(pv, dem, pz, tip, e_nom=C.E_NOM, soc0=C.SOC0, timelimit=None):
    """MILP of Section 2.2: hourly energy balance, shared energy as min(injection, withdrawal) through two upper
    bounds, SOC dynamics, binary charge/discharge exclusivity, no grid charging, RID on the whole injection."""
    T = len(pv); H = range(T)
    m = pulp.LpProblem("REC", pulp.LpMaximize)
    E = pulp.LpVariable.dicts("E", H, C.SOC_MIN * e_nom, C.SOC_MAX * e_nom)
    Pc = pulp.LpVariable.dicts("Pc", H, 0, C.P_MAX); Pd = pulp.LpVariable.dicts("Pd", H, 0, C.P_MAX)
    Pi = pulp.LpVariable.dicts("Pi", H, 0); Es = pulp.LpVariable.dicts("Es", H, 0)
    u = pulp.LpVariable.dicts("u", H, cat="Binary")
    m += pulp.lpSum(float((tip[h] + C.ARERA_VALORIZATION) / 1e3) * Es[h] + float(pz[h] / 1e3) * (Pi[h] + Pd[h])
                    for h in H)
    for h in H:
        m += float(pv[h]) == Pc[h] + Pi[h]
        m += E[h] == (soc0 * e_nom if h == 0 else E[h - 1]) + float(C.ETA) * Pc[h] - float(1 / C.ETA) * Pd[h]
        m += Pc[h] <= C.P_MAX * u[h]
        m += Pd[h] <= C.P_MAX * (1 - u[h])
        m += Es[h] <= float(dem[h])
        m += Es[h] <= Pi[h] + Pd[h]
    t0 = time.time()
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timelimit))
    dt = time.time() - t0
    pc = np.array([Pc[h].varValue for h in H]); pd_ = np.array([Pd[h].varValue for h in H])
    es = np.array([Es[h].varValue for h in H])
    return dict(pc=pc, pd=pd_, es=es, status=pulp.LpStatus[m.status], time=dt,
                nvar=len(m.variables()), ncon=len(m.constraints), nbin=T)


def rule(pv, dem, pz, mode, e_nom=C.E_NOM):
    """Rule-based self-consumption ('sc') and price-driven arbitrage ('arb') strategies of Section 2.2.

    The series start at 00:00, so that the hour of the day of step t is t mod 24.
    """
    soc = C.SOC0 * e_nom; T = len(pv); pc = np.zeros(T); pd_ = np.zeros(T)
    for t in range(T):
        if mode == "sc":
            ch = min(max(0.0, pv[t] - dem[t]), C.P_MAX, (C.SOC_MAX * e_nom - soc) / C.ETA)
            dis = min(max(0.0, dem[t] - pv[t]), C.P_MAX, (soc - C.SOC_MIN * e_nom) * C.ETA)
        else:
            h = t % 24
            d = 1 if (pz[t] >= C.ARB_PRICE_ANY_HOUR or (pz[t] >= C.ARB_PRICE_EVENING and h >= C.ARB_EVENING_HOUR)) \
                else 0
            dis = d * min(C.P_MAX, (soc - C.SOC_MIN * e_nom) * C.ETA)
            ch = min(pv[t], C.P_MAX, (C.SOC_MAX * e_nom - soc) / C.ETA)
            if ch > 0:
                dis = 0.0
        soc += ch * C.ETA - dis / C.ETA
        pc[t], pd_[t] = ch, dis
    return dict(pc=pc, pd=pd_)


def econ(inj, dem, pz, tip):
    """Shared energy and revenue streams (incentive, ARERA valorization, RID on the whole injection) [€]."""
    sh = np.minimum(inj, dem)
    inc = (sh * tip).sum() / 1e3; ar = (sh * C.ARERA_VALORIZATION).sum() / 1e3; rid = (inj * pz).sum() / 1e3
    return dict(inj=inj, sh=sh, shared_MWh=sh.sum() / 1e3, inj_MWh=inj.sum() / 1e3, inc=inc, arera=ar, rid=rid,
                gross=inc + ar + rid, rho=100 * sh.sum() / inj.sum())


def realize(pv, dem, pz, tip, pc_plan, pd_plan, e_nom=C.E_NOM, soc0=C.SOC0):
    """Open-loop application of a dispatch plan to the measured load, with the physical SOC limits enforced."""
    soc = soc0 * e_nom; T = len(pv); inj = np.zeros(T); pd_r = np.zeros(T)
    for t in range(T):
        dis = min(pd_plan[t], (soc - C.SOC_MIN * e_nom) * C.ETA)
        ch = min(pc_plan[t], (C.SOC_MAX * e_nom - soc) / C.ETA, pv[t])
        inj[t] = max(0.0, pv[t] - ch + dis); soc += ch * C.ETA - dis / C.ETA; pd_r[t] = dis
    return econ(inj, dem, pz, tip) | {"discharge": pd_r}
