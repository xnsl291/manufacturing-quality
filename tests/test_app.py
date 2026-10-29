####################################################################
# test_app.py — 화면 세 페이지가 예외 없이 뜨는지 확인함
#
# 페이지 1·3 은 저장소에 커밋된 집계표만으로 떠야 한다 (원본 데이터 없이
# clone 만 한 사람도 볼 수 있어야 함). 페이지 2 는 로컬 산출물이 있으면
# 목록을, 없으면 안내 문구를 띄워야 한다. 두 경우를 PC 상태와 상관없이
# 모두 확인하려고 산출물 경로를 테스트에서 바꿔 끼운다.
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import common  # noqa: E402

PAGES = ROOT / "app" / "views"
REAL_ARTIFACT = common.ARTIFACT


def run(page: str) -> AppTest:
    return AppTest.from_file(str(PAGES / page), default_timeout=180).run()


def fake_parts(path: Path, n: int = 1000) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.05).astype("int8")
    pd.DataFrame({
        "Id": np.arange(n),
        "start_time": np.linspace(0, 10, n),
        "duration": rng.random(n),
        "visited_station_count": rng.integers(5, 20, n),
        "visited_line_count": rng.integers(1, 4, n),
        "measurement_count": rng.integers(50, 300, n),
        "path_id": rng.integers(0, 3, n).astype("int16"),
        "위험도": rng.random(n) + y,
        "실제_불량": y,
        "방문_공정": ["L0_S0|L3_S29"] * n,
    }).to_parquet(path, index=False)


def test_overview_shows_headline_numbers():
    at = run("overview.py")
    assert not at.exception
    values = [m.value for m in at.metric]
    # 기본 검사 물량 1% 에서 README 헤드라인과 같은 숫자가 나와야 함
    assert "11.5%" in values
    assert any("11.5배" in (m.proto.delta or "") for m in at.metric)


def test_overview_slider_moves_kpi():
    at = run("overview.py")
    before = at.metric[3].value
    at.select_slider[0].set_value(0.05).run()
    assert not at.exception
    assert at.metric[3].value != before


def test_diagnostics_renders():
    at = run("diagnostics.py")
    assert not at.exception
    assert len(at.dataframe) >= 5


def test_diagnostics_shows_readable_units():
    # 2026-10-29 표기 개선: 비율은 %, Lift 는 배로 읽히게 바꿨다. 소수 그대로 두면
    # 보는 사람이 0.1151 을 머릿속에서 11.5% 로 환산해야 한다
    at = run("diagnostics.py")
    assert not at.exception
    comp = at.dataframe[0].value
    assert comp.loc["2_LightGBM", "포착률@1%"] == "11.5%"
    assert comp.loc["2_LightGBM", "무작위 대비@1%"] == "11.5배"
    # 누설 포함 행은 지우지 않고 남긴다 — 배제 대상이라는 것 자체가 보여 줄 내용임
    assert any(str(i).startswith("X_") for i in comp.index)
    # 기여도 표의 열 이름은 코드 이름이 아니라 화면용 한국어여야 함
    contrib = at.dataframe[1].value
    assert "공정" in contrib.columns
    assert "세 방식 중 가장 낮은 순위" in contrib.columns
    assert "station_key" not in contrib.columns
    # 시기별 포착률은 표와 막대를 같이 둔다 (표 2 의 비중과 헷갈리지 않게)
    assert len(at.get("vega_lite_chart")) == 1


def test_risk_monitor_without_artifact_shows_guide(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACT", tmp_path / "없음.parquet")
    at = run("risk_monitor.py")
    assert not at.exception
    assert any("build_artifacts" in i.value for i in at.info)
    assert len(at.dataframe) == 0


def test_risk_monitor_with_artifact_lists_top_rate(tmp_path, monkeypatch):
    path = tmp_path / "parts.parquet"
    fake_parts(path)
    monkeypatch.setattr(common, "ARTIFACT", path)
    at = run("risk_monitor.py")
    assert not at.exception
    assert at.metric[0].value == "10대"  # 1000대의 기본 1%
    assert len(at.dataframe) == 1


@pytest.mark.skipif(not REAL_ARTIFACT.exists(), reason="로컬 산출물이 있는 PC 에서만 확인 가능")
def test_risk_monitor_matches_overview_at_one_percent():
    # 같은 1% 에서 페이지 1(집계표)과 페이지 2(제품 단위)가 같은 포착률을 내야 함
    overview = run("overview.py")
    monitor = run("risk_monitor.py")
    assert overview.metric[3].value == monitor.metric[2].value
