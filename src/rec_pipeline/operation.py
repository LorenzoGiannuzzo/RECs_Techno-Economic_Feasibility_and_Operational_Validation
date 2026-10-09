"""Step 3 - operation track: rolling-origin validation of the MLP forecaster, naive benchmarks, peak-timing error,
out-of-sample SHAP, and EMS robustness test (Sections 2.4 and 2.5, results in Sections 4.3 and 4.4)."""
import json
import time

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score

from . import config as C
from .forecasting import FEATURES, build, make_mlp
from .models import econ, milp, realize

NON_RES = ["Public administration", "Industrial", "Street lighting"]


def run(inp, paths, log=print):
    out = paths.results_dir
    R = {}
    F = build(inp)
    ts, y_raw, X_raw, r_raw, day_mean, valid = F.ts, F.y, F.X, F.r, F.day_mean, F.valid
    R["res_check_MWh"] = dict(sample=y_raw.sum() / 1e3, scaled=y_raw.sum() * C.K_SCALE / 1e3,
                              sim=inp.cats_chrono["Residential"].sum() / 1e3)
    R["n_train_lost_lags"] = int((~valid).sum())

    #Lorenzo Giannuzzo: rolling-origin validation with an expanding window: October-December 2024 form the initial
    # training window, and the model is retrained at the beginning of each month from January to September 2025 using
    # only the data recorded before that date; inputs and target are standardized on the training window only
    log(">>> Rolling-origin validation of the MLP forecaster")
    months = pd.period_range("2025-01", "2025-09", freq="M")
    pred = np.full(C.HOURS, np.nan); fold_info = []; shap_rows = []; shap_X = []
    rng = np.random.default_rng(C.SEED)
    t_all = time.time()
    for per in months:
        start = per.start_time; end = (per + 1).start_time
        tr = (ts < start) & valid; te = (ts >= start) & (ts < end)
        mu, sd = X_raw[tr].mean(0), X_raw[tr].std(0) + 1e-9
        X = (X_raw - mu) / sd
        rm, rs = r_raw[tr].mean(), r_raw[tr].std()
        mlp = make_mlp()
        t0 = time.time(); mlp.fit(X[tr], (r_raw[tr] - rm) / rs); dt = time.time() - t0

        def ratio(Z, mlp=mlp, rm=rm, rs=rs):
            return mlp.predict(Z) * rs + rm

        pred[te] = ratio(X[te]) * day_mean[te]
        fold_info.append(dict(month=str(per), n_train=int(tr.sum()), n_test=int(te.sum()), train_time_s=dt,
                              epochs=int(mlp.n_iter_),
                              r2_train=float(r2_score(y_raw[tr], ratio(X[tr]) * day_mean[tr])),
                              r2_test=float(r2_score(y_raw[te], pred[te]))))
        #Lorenzo Giannuzzo: kernel SHAP of the predicted ratio on 34 out-of-sample hours of the fold; since the mean
        # load of the previous day multiplies the whole prediction, the values are converted into kWh of community demand
        bg = shap.kmeans(X[tr], 50)
        expl = shap.KernelExplainer(ratio, bg)
        idx = rng.choice(np.where(te)[0], 34, replace=False)
        sv = expl.shap_values(X[idx], silent=True) * (day_mean[idx] * C.K_SCALE)[:, None]
        shap_rows.append(sv); shap_X.append(X_raw[idx])
        log(f"    {per}: out-of-sample R2 = {fold_info[-1]['r2_test']:.3f} (training {dt:.1f} s)")
    R["folds"] = fold_info
    R["train_total_s"] = time.time() - t_all

    oos = ~np.isnan(pred)
    yt, yp = y_raw[oos] * C.K_SCALE, pred[oos] * C.K_SCALE
    res = yt - yp
    R["mlp_oos"] = dict(r2=float(r2_score(yt, yp)), rmse=float(np.sqrt((res ** 2).mean())),
                        mae=float(np.abs(res).mean()), mape=float(100 * np.mean(np.abs(res) / yt)),
                        mean_load=float(yt.mean()), n=int(oos.sum()),
                        r2_train_mean=float(np.mean([f["r2_train"] for f in fold_info])))
    #Lorenzo Giannuzzo: naive benchmarks on the same hours (same hour of the previous week and of the previous day)
    for key, k in [("naive_res_oos", 168), ("naive24_res_oos", 24)]:
        nv = np.r_[y_raw[:k], y_raw[:-k]] * C.K_SCALE
        R[key] = dict(r2=float(r2_score(yt, nv[oos])), rmse=float(np.sqrt(((yt - nv[oos]) ** 2).mean())),
                      mape=float(100 * np.mean(np.abs(yt - nv[oos]) / yt)))
    R["mlp_oos_monthly_r2"] = {f["month"]: f["r2_test"] for f in fold_info}

    #Lorenzo Giannuzzo: peak-timing error on a circular 24-hour clock (Eq. 42), over the whole day and over the hours
    # from 16:00 onward, in which the BESS discharges
    dd = pd.DataFrame({"t": ts[oos], "y": yt, "p": yp}); dd["day"] = dd.t.dt.date; dd["h"] = dd.t.dt.hour
    hy = dd.groupby("day").apply(lambda g: g.h.values[g.y.values.argmax()])
    hp = dd.groupby("day").apply(lambda g: g.h.values[g.p.values.argmax()])
    e = np.abs(hy - hp); e = np.minimum(e, 24 - e)
    R["peak_timing"] = dict(mean_h=float(e.mean()), within1=float(100 * (e <= 1).mean()),
                            within2=float(100 * (e <= 2).mean()), days=int(len(e)))
    ev = dd[dd.h >= 16]
    hy2 = ev.groupby("day").apply(lambda g: g.h.values[g.y.values.argmax()])
    hp2 = ev.groupby("day").apply(lambda g: g.h.values[g.p.values.argmax()])
    e2 = np.abs(hy2 - hp2)
    R["peak_timing_evening"] = dict(mean_h=float(e2.mean()), within1=float(100 * (e2 <= 1).mean()))
    #Lorenzo Giannuzzo: same metric for the persistence of the previous day, as a benchmark
    nv = (np.r_[y_raw[:24], y_raw[:-24]] * C.K_SCALE)[oos]
    dn = pd.DataFrame({"t": ts[oos], "y": yt, "p": nv}); dn["day"] = dn.t.dt.date; dn["h"] = dn.t.dt.hour
    hn = dn.groupby("day").apply(lambda g: g.h.values[g.p.values.argmax()])
    en = np.abs(hy - hn); en = np.minimum(en, 24 - en)
    evn = dn[dn.h >= 16]
    hn2 = evn.groupby("day").apply(lambda g: g.h.values[g.p.values.argmax()])
    R["peak_timing_naive24"] = dict(mean_h=float(en.mean()), within1=float(100 * (en <= 1).mean()),
                                    evening_mean_h=float(np.abs(hy2 - hn2).mean()))

    S = np.vstack(shap_rows); SX = np.vstack(shap_X)
    R["shap"] = {f: float(np.abs(S[:, i]).mean()) for i, f in enumerate(FEATURES)}
    np.savez(out / "forecast.npz", ts=ts.values.astype("int64"), y=y_raw * C.K_SCALE, pred=pred * C.K_SCALE,
             shap=S, shapX=SX)

    #Lorenzo Giannuzzo: EMS robustness test on the out-of-sample hours (January-September 2025, positions 0-6551 of
    # the simulated year): perfect-foresight MILP over the whole period against day-ahead plans applied open-loop to
    # the measured loads, with the residential load forecast by the MLP and the non-residential loads measured (A) or
    # forecast by the seasonal-naive predictor (B), and with all users forecast by the seasonal-naive predictor (C);
    # PV generation and day-ahead zonal prices are taken as known
    log(">>> EMS robustness test")
    T = 6552
    assert np.allclose(inp.cats["Residential"][:T], inp.cats_chrono["Residential"][2208:])
    res_fc = (pred * C.K_SCALE)[2208:]
    week = lambda v: np.r_[v[:168], v[:-168]]
    nonres_meas = sum(inp.cats[k][:T] for k in NON_RES)
    nonres_naive = sum(week(inp.cats_chrono[k])[2208:] for k in NON_RES)
    res_naive = week(inp.cats_chrono["Residential"])[2208:]
    R["nonres_naive_r2"] = {k: float(r2_score(inp.cats_chrono[k][2208:], week(inp.cats_chrono[k])[2208:]))
                            for k in NON_RES}
    pv, dem, pz, tip = inp.pv[:T], inp.dem[:T], inp.pz[:T], inp.tip[:T]
    pf = milp(pv, dem, pz, tip)
    pf_e = econ(pv - pf["pc"] + pf["pd"], dem, pz, tip)
    R["pf"] = dict(gross=pf_e["gross"], shared=pf_e["shared_MWh"], time=pf["time"])
    nob = econ(pv.copy(), dem, pz, tip)
    R["nobess_gross"] = nob["gross"]
    plans = {"A": res_fc + nonres_meas, "B": res_fc + nonres_naive, "C": res_naive + nonres_naive}
    R["plans"] = {}
    mon = inp.ts_sim[:T].month
    for k, dem_fc in plans.items():
        #Lorenzo Giannuzzo: day-ahead planning: at the end of each day the MILP is solved over the 24 hours of the next
        # day on the forecast demand, with the day-ahead zonal prices and the SOC actually reached, and the plan is
        # then applied open-loop to the measured demand of that day
        soc = C.SOC0 * C.E_NOM; inj = np.zeros(T); pc_p = np.zeros(T); pd_p = np.zeros(T); expected = 0.0
        for d in range(T // 24):
            a, b = 24 * d, 24 * (d + 1)
            pl = milp(pv[a:b], dem_fc[a:b], pz[a:b], tip[a:b], soc0=soc / C.E_NOM)
            pc_p[a:b], pd_p[a:b] = pl["pc"], pl["pd"]
            expected += econ(pv[a:b] - pl["pc"] + pl["pd"], dem_fc[a:b], pz[a:b], tip[a:b])["gross"]
            day = realize(pv[a:b], dem[a:b], pz[a:b], tip[a:b], pl["pc"], pl["pd"], soc0=soc / C.E_NOM)
            inj[a:b] = day["inj"]
            soc = day["soc_end"]
        real = econ(inj, dem, pz, tip)
        R["plans"][k] = dict(expected=expected, realized=real["gross"], shared=real["shared_MWh"],
                             gap=100 * (pf_e["gross"] - real["gross"]) / pf_e["gross"],
                             gap_expected=100 * (pf_e["gross"] - expected) / pf_e["gross"],
                             gap_storage_value=100 * (pf_e["gross"] - real["gross"]) / (pf_e["gross"] - nob["gross"]),
                             demand_r2=float(r2_score(dem, dem_fc)),
                             demand_mape=float(100 * np.mean(np.abs(dem - dem_fc) / dem)))
        hr_pf = pf_e["sh"] * (tip + C.ARERA_VALORIZATION) / 1e3 + pf_e["inj"] * pz / 1e3
        hr_rl = real["sh"] * (tip + C.ARERA_VALORIZATION) / 1e3 + real["inj"] * pz / 1e3
        g = pd.DataFrame({"m": mon, "p": hr_pf, "r": hr_rl}).groupby("m").sum()
        R["plans"][k]["monthly_gap"] = (100 * (g.p - g.r) / g.p).round(3).tolist()
        np.savez(out / f"plan_{k}.npz", pf_sh=pf_e["sh"], real_sh=real["sh"], hr_pf=hr_pf, hr_rl=hr_rl)
        log(f"    plan {k}: revenue gap = {R['plans'][k]['gap']:.2f}%")
    inj_pf = pf_e["inj"]; pvh = pv > 0
    R["pf_demand_limited"] = float(100 * ((dem >= inj_pf) & pvh).sum() / pvh.sum())
    json.dump(R, open(out / "results_ops.json", "w"), indent=1, default=float)
    log(">>> Step 3 (operation) done")
    return R
