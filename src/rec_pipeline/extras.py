"""Step 4 - complementary quantities quoted in the text: random-split benchmark of the forecaster, solve time of the
24-hour MILP, revenue gap of the critical week, present value of the MILP revenue increment, and price statistics
behind the arbitrage thresholds."""
import json
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from . import config as C
from .forecasting import build, make_mlp
from .models import milp


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(inp, paths, log=print):
    out = paths.results_dir
    X = {}
    #Lorenzo Giannuzzo: random 70/30 split of the hourly samples with the same inputs, target, and hyperparameters as
    # the rolling-origin forecaster, which quantifies the optimistic bias of a random partition
    F = build(inp)
    ok = np.where(F.valid)[0]
    idx_tr, idx_te = train_test_split(ok, test_size=0.3, random_state=C.SEED)
    mu, sd = F.X[idx_tr].mean(0), F.X[idx_tr].std(0) + 1e-9; Xs = (F.X - mu) / sd
    rm, rs = F.r[idx_tr].mean(), F.r[idx_tr].std()
    mlp = make_mlp().fit(Xs[idx_tr], (F.r[idx_tr] - rm) / rs)
    fc = lambda i: (mlp.predict(Xs[i]) * rs + rm) * F.day_mean[i]
    X["random_split_r2_test"] = float(r2_score(F.y[idx_te], fc(idx_te)))
    X["random_split_r2_train"] = float(r2_score(F.y[idx_tr], fc(idx_tr)))

    #Lorenzo Giannuzzo: solve time of the 24-hour MILP of a receding-horizon implementation (four sample days)
    times = []
    for day in [30, 120, 200, 300]:
        s = slice(24 * day, 24 * day + 24)
        times.append(milp(inp.pv[s], inp.dem[s], inp.pz[s], inp.tip[s])["time"])
    X["milp24_time_mean_s"] = float(np.mean(times)); X["milp24_time_max_s"] = float(np.max(times))

    #Lorenzo Giannuzzo: revenue gap of the critical week of August and of the weeks of August (plan B)
    B = np.load(out / "plan_B.npz"); t = pd.date_range(C.SIM_YEAR_START, periods=len(B["hr_pf"]), freq="h")
    w = (t >= "2025-08-15") & (t < "2025-08-22")
    X["week_rev_gap_pct"] = float(100 * (B["hr_pf"][w].sum() - B["hr_rl"][w].sum()) / B["hr_pf"][w].sum())
    aug = pd.DataFrame({"p": B["hr_pf"], "r": B["hr_rl"]}, index=t); aug = aug[aug.index.month == 8]
    wk = aug.groupby(pd.Grouper(freq="7D", origin=pd.Timestamp("2025-08-01"))).sum()
    X["aug_weekly_gaps"] = (100 * (wk.p - wk.r) / wk.p).round(2).tolist()

    #Lorenzo Giannuzzo: present value of the RID increment of MILP over the baseline against the additional investment
    # and operating cost of the battery (Section 5.3)
    R = json.load(open(out / "results_design.json"))
    disc = (1 + C.RATE) ** np.arange(1, C.N_YEARS + 1)
    inc = np.array([a["rid"] - b["rid"] for a, b in zip(R["yearly"]["MILP"], R["yearly"]["No-BESS"])])
    X["rid_inc_y1"] = float(inc[0]); X["rid_inc_pv"] = float((inc / disc).sum())
    X["extra_opex_pv"] = float(((R["dcf"]["MILP"]["opex"] - R["dcf"]["No-BESS"]["opex"]) / disc).sum())
    X["burden"] = C.E_NOM * C.C_BESS + X["extra_opex_pv"]
    X["cpu"] = _cpu_name()

    #Lorenzo Giannuzzo: statistics of the 2025 South-zone prices behind the arbitrage thresholds of Eq. 6
    pz = inp.pz; hrs = np.arange(C.HOURS) % 24
    X["p125_share"] = float(100 * (pz >= C.ARB_PRICE_ANY_HOUR).mean())
    X["p125_pct"] = float((pz < C.ARB_PRICE_ANY_HOUR).mean() * 100)
    X["eve85_share"] = float(100 * (pz[hrs >= C.ARB_EVENING_HOUR] >= C.ARB_PRICE_EVENING).mean())
    X["q75"] = float(np.percentile(pz, 75)); X["q_eve_25"] = float(np.percentile(pz[hrs >= C.ARB_EVENING_HOUR], 25))
    X["eve_median"] = float(np.median(pz[hrs >= C.ARB_EVENING_HOUR]))
    X["day_median"] = float(np.median(pz[hrs < C.ARB_EVENING_HOUR]))
    json.dump(X, open(out / "extra.json", "w"), indent=1)
    log(">>> Step 4 (complementary quantities) done")
    return X
