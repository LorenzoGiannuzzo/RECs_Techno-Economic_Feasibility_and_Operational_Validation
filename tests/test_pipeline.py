"""Fast tests of the data interface and of the dispatch models (run with: python -m pytest)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rec_pipeline import config as C  # noqa: E402
from rec_pipeline.data import DataError, load_inputs, tip_tariff  # noqa: E402
from rec_pipeline.models import econ, milp, realize, rule  # noqa: E402
from rec_pipeline.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def inputs(tmp_path_factory):
    folder = generate(tmp_path_factory.mktemp("synthetic"))
    return load_inputs(folder)


def test_tip_tariff():
    assert np.allclose(tip_tariff(np.array([50.0, 150.0, 179.0, 180.0, 250.0])), [100.0, 90.0, 61.0, 60.0, 60.0])


def test_inputs_shape_and_calendar(inputs):
    assert len(inputs.pv) == len(inputs.pz) == len(inputs.dem) == C.HOURS
    assert set(inputs.cats) == {"Residential", "Public administration", "Industrial", "Street lighting"}
    assert inputs.ts_sim[0].dayofweek == 2 and inputs.ts_meas[0].dayofweek == 1
    assert np.isfinite(inputs.dem).all() and (inputs.dem >= 0).all()


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(DataError, match="missing input file"):
        load_inputs(tmp_path)


def test_milp_two_days(inputs):
    s = slice(24 * 180, 24 * 182)
    pv, dem, pz, tip = inputs.pv[s], inputs.dem[s], inputs.pz[s], inputs.tip[s]
    m = milp(pv, dem, pz, tip)
    assert m["status"] == "Optimal"
    assert not ((m["pc"] > 1e-6) & (m["pd"] > 1e-6)).any()
    inj = pv - m["pc"] + m["pd"]
    assert np.allclose(m["es"], np.minimum(inj, dem), atol=1e-4)
    base = econ(pv, dem, pz, tip)["gross"]
    assert econ(inj, dem, pz, tip)["gross"] >= base - 1e-6


@pytest.mark.parametrize("mode", ["sc", "arb"])
def test_rule_respects_soc_window(inputs, mode):
    s = slice(0, 24 * 7)
    r = rule(inputs.pv[s], inputs.dem[s], inputs.pz[s], mode)
    soc = C.SOC0 * C.E_NOM + np.cumsum(r["pc"] * C.ETA - r["pd"] / C.ETA)
    assert soc.min() >= C.SOC_MIN * C.E_NOM - 1e-6 and soc.max() <= C.SOC_MAX * C.E_NOM + 1e-6
    assert (r["pc"] <= inputs.pv[s] + 1e-9).all()


def test_open_loop_realization_is_feasible(inputs):
    s = slice(24 * 100, 24 * 103)
    pv, dem, pz, tip = inputs.pv[s], inputs.dem[s], inputs.pz[s], inputs.tip[s]
    plan = milp(pv, dem * 1.2, pz, tip)
    real = realize(pv, dem, pz, tip, plan["pc"], plan["pd"])
    assert (real["sh"] <= dem + 1e-9).all() and (real["sh"] <= real["inj"] + 1e-9).all()
