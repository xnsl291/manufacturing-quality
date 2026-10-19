####################################################################
# test_figures.py — 그림용 집계와 그림 함수를 검증함
#
# 특히 노리는 것
#   ① 촘촘한 곡선이 대표 표(threshold_table)와 같은 점에서 같은 숫자를
#      내는가 — 어긋나면 그림과 README 헤드라인이 다른 말을 함
#   ② 경로 방문 비율 표가 제품 단위 행 없이 경로 묶음 단위로만 나오는가
#   ③ 저장소의 집계표만으로 그림 11장이 전부 그려지는가 (원본 불필요)
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import figures  # noqa: E402
from src.build_artifacts import (  # noqa: E402
    CURVE_RATES,
    capture_curve,
    path_station_visit,
    scored_parts,
    start_time_hist,
)
from src.metrics import threshold_table  # noqa: E402


def fake_feats(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    path_id = rng.choice([0, 1, 2], size=n)
    f = pd.DataFrame({
        "Id": np.arange(n),
        "visited_L0_S0": (path_id == 1).astype("int8"),
        "visited_L3_S29": np.where(path_id == 0, rng.integers(0, 2, n), 1).astype("int8"),
        "visited_station_count": rng.integers(5, 20, n),
        "visited_line_count": rng.integers(1, 4, n),
        "measurement_count": rng.integers(50, 300, n),
        "start_time": np.linspace(0, 100, n),
        "duration": rng.random(n),
        "path_id": path_id.astype("int16"),
        "split": np.array(["train"] * (n // 2) + ["valid"] * (n // 4) + ["test"] * (n - n // 2 - n // 4)),
    })
    y = (rng.random(n) < 0.1).astype(int)
    return f, y


def test_curve_contains_headline_rates_and_matches_table():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    s = rng.random(5000) + y * 0.5
    curve = capture_curve(y, {"m": s})
    head = threshold_table(y, s)
    for rate in (0.005, 0.01, 0.03, 0.05, 0.10):
        assert rate in CURVE_RATES
        a = curve.loc[curve["목표_검사율"] == rate, "포착률"].iloc[0]
        b = head.loc[head["목표_검사율"] == rate, "포착률"].iloc[0]
        assert a == b
    assert (curve["목표_검사율"] < 1.0).all()  # 전수 검사 행은 곡선에서 뺌


def test_curve_perfect_model_hits_theoretical_recall():
    # threshold_table 과 비교하는 것만으로는 둘이 같이 틀려도 통과함.
    # 손으로 아는 정답과도 맞춰 본다: 완벽한 모델은 검사율 a 에서 min(1, a/불량률) 을 잡음
    y = np.zeros(10000, dtype=int)
    y[:200] = 1  # 불량률 2%
    s = y + np.linspace(0, 0.5, len(y))  # 불량이 항상 위, 동점 없음
    curve = capture_curve(y, {"perfect": s})
    for rate in (0.005, 0.01, 0.03):
        got = curve.loc[curve["목표_검사율"] == rate, "포착률"].iloc[0]
        assert got == pytest.approx(min(1.0, rate / 0.02), abs=1e-3)


def test_path_visit_is_group_level_only():
    f, y = fake_feats()
    t = path_station_visit(f, y)
    assert len(t) == f["path_id"].nunique()  # 경로 묶음 수만큼, 제품 수가 아님
    assert t["제품수"].sum() == len(f)
    assert t.loc["경로 01", "L0_S0"] == 1.0
    assert t.loc["경로 02", "L0_S0"] == 0.0
    assert 0 < t.loc["기타(희귀 경로)", "L3_S29"] < 1


def test_start_time_hist_counts_everything_and_marks_split():
    f, y = fake_feats()
    f.loc[3, "start_time"] = np.nan  # 날짜 없는 제품은 셀 수 없음
    h = start_time_hist(f, y, bins=10)
    assert h["제품수"].sum() == len(f) - 1
    assert h["불량수"].sum() == y[f["start_time"].notna()].sum()
    assert h["시험_시작"].iloc[0] > h["검증_시작"].iloc[0]


def test_scored_parts_only_test_split():
    f, y = fake_feats()
    te = (f["split"] == "test").to_numpy()
    out = scored_parts(f, y, np.arange(te.sum(), dtype=float))
    assert len(out) == te.sum()
    assert (out["실제_불량"].to_numpy() == y[te]).all()
    assert out.loc[out["Id"] == f.loc[te, "Id"].iloc[0], "방문_공정"].notna().all()


def test_load_missing_table_names_the_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_artifacts"):
        figures.load("T_포착률곡선", tmp_path)


def test_inflation_table_matches_readme():
    t = figures.inflation_table()
    assert t["정직한_조건_대비"].round(1).tolist() == [1.0, 2.2, 9.0, 9.4]


def test_f3_uses_same_path_numbers_as_f2():
    # F2·F3·위험 모니터가 같은 "경로 NN" 을 같은 경로로 가리켜야 함 (독립 검토 지적)
    visit = figures.load("T_경로x공정_방문비율")
    fig = figures.f3_paths(visit)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels[-1].startswith("경로 01")
    assert not any(l.startswith("기타") for l in labels)
    plt.close(fig)


def test_all_figures_draw_from_committed_tables():
    figs = figures.build_all()
    assert len(figs) == 11
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
