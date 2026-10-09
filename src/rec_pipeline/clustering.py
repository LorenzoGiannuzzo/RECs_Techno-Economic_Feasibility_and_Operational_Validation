"""Step 2 - characterization of the residential demand: hierarchical clustering of the normalized daily profiles and
random forest day-type classifier (Section 2.4, results in Section 4.3, Table 8, and Fig. 7)."""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, davies_bouldin_score, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split

from . import config as C

FEATURE_LABELS = {"T_avg": "Mean daily temperature", "T_max": "Maximum daily temperature",
                  "T_min": "Minimum daily temperature", "day_Mon": "Monday indicator", "day_Sat": "Saturday indicator",
                  "day_Sun": "Sunday indicator", "day_Thu": "Thursday indicator", "day_Tue": "Tuesday indicator",
                  "day_Wed": "Wednesday indicator"}


def run(inp, paths, log=print):
    out = paths.results_dir
    R = {}
    #Lorenzo Giannuzzo: daily profiles of the residential sample (zeros treated as missing and interpolated),
    # normalized by their daily maximum
    e = pd.Series(inp.res_sample_clustering.copy())
    e[e == 0] = np.nan
    e = e.interpolate().values
    df = pd.DataFrame({"date": inp.ts_meas.normalize(), "hour": inp.ts_meas.hour, "e": e})
    M = df.pivot(index="date", columns="hour", values="e")
    M = M.div(M.max(axis=1), axis=0)
    V = M.values

    #Lorenzo Giannuzzo: internal validity indices used to select the number of clusters (Eqs. 43-44)
    log(">>> Hierarchical clustering")
    R["validity"] = {}
    for k in C.CLUSTER_K_RANGE:
        lab_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(V)
        R["validity"][k] = dict(silhouette=float(silhouette_score(V, lab_k)),
                                davies_bouldin=float(davies_bouldin_score(V, lab_k)))
    lab = AgglomerativeClustering(n_clusters=C.CLUSTER_K, linkage="ward").fit_predict(V)
    cent = {i: V[lab == i].mean(0) for i in range(C.CLUSTER_K)}
    rmse = {i: float(np.sqrt(np.mean(np.mean((V[lab == i] - cent[i]) ** 2, axis=1)))) for i in range(C.CLUSTER_K)}
    dates = pd.to_datetime(M.index)
    hol = dates.isin(pd.to_datetime(C.HOLIDAYS))
    R["clusters"] = {}
    for i in range(C.CLUSTER_K):
        sel = lab == i
        R["clusters"][i + 1] = dict(days=int(sel.sum()), rmse=rmse[i], peak_hour=int(np.argmax(cent[i])),
                                    by_weekday={d: int(((dates.dayofweek == j) & sel).sum()) for j, d in
                                                enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])},
                                    by_month={int(m): int(((dates.month == m) & sel).sum()) for m in range(1, 13)},
                                    holidays=int((hol & sel).sum()))
    log("    " + ", ".join(f"cluster {k}: {v['days']} days, RMSE {v['rmse']:.3f}"
                          for k, v in R["clusters"].items()))

    #Lorenzo Giannuzzo: random forest day-type classifier on the daily temperatures and on the day of the week
    # (one-hot encoding with the first category dropped), 80/20 random split of the days (Eq. 32)
    log(">>> Random forest day-type classifier")
    wd = inp.weather_daily.set_index("date").reindex(dates)
    data = pd.DataFrame({"T_avg": wd.t_avg_c.values, "T_max": wd.t_max_c.values, "T_min": wd.t_min_c.values,
                         "day": dates.day_name().str[:3]})
    X = pd.get_dummies(data, columns=["day"], drop_first=True)
    y = pd.Series(lab)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=C.SEED)
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=C.SEED).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    e2e = np.mean([np.sqrt(np.mean((V[i] - cent[p]) ** 2)) for i, p in zip(X_te.index, y_pred)])
    cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    R["classifier"] = dict(accuracy=100 * accuracy_score(y_te, y_pred),
                           f1_macro=100 * f1_score(y_te, y_pred, average="macro"),
                           recall_macro=100 * recall_score(y_te, y_pred, average="macro"),
                           precision_macro=100 * precision_score(y_te, y_pred, average="macro"),
                           cv5_accuracy_mean=float(100 * cv.mean()), cv5_accuracy_std=float(100 * cv.std()),
                           end_to_end_rmse=float(e2e), n_test=int(len(y_te)),
                           confusion_matrix=confusion_matrix(y_te, y_pred).tolist())
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    R["feature_importance"] = {FEATURE_LABELS.get(k, k): float(v) for k, v in imp.items()}
    log(f"    accuracy {R['classifier']['accuracy']:.2f}%, macro F1 {R['classifier']['f1_macro']:.2f}%, "
        f"end-to-end RMSE {R['classifier']['end_to_end_rmse']:.4f}")
    np.savez(out / "clusters.npz", profiles=V, labels=lab, rmse=np.array([rmse[i] for i in range(C.CLUSTER_K)]))
    json.dump(R, open(out / "results_clustering.json", "w"), indent=1, default=float)
    log(">>> Step 2 (clustering and classification) done")
    return R
