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

import re
import tomllib

import numpy as np
import pandas as pd
import pytest
from streamlit import config as st_config
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import common  # noqa: E402

REAL_ARTIFACT = common.ARTIFACT
CONFIG = ROOT / "app" / ".streamlit" / "config.toml"


@pytest.fixture(autouse=True)
def app_config():
    # AppTest 는 app/.streamlit/config.toml 을 읽지 않음. 실제 실행과 같은 값을 넣어 2번 페이지의
    # 설정 확인 장치를 통과시킴 (값 자체는 test_config_keeps_page2_local 이 파일에서 확인함)
    for k, v in (("server.address", "localhost"), ("client.disableDataExport", True)):
        st_config.set_option(k, v)
    yield


def run(page: str, **state) -> AppTest:
    # 페이지끼리 링크(st.page_link·switch_page)가 있어 메뉴를 등록하는 main.py 를 거쳐 띄움
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=180).run()
    for k, v in state.items():
        at.session_state[k] = v
    return at.switch_page(f"views/{page}").run()


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


def test_overview_slider_ticks_are_readable():
    # 2026-10-29: 곡선 표 65칸을 그대로 쓰면 0.500% 와 0.504% 가 둘 다 "0.50%" 로 보였다.
    # 운영에서 고를 만한 7칸만 남기고 끝자리 0 을 떼어 읽기 쉽게 한다
    at = run("overview.py")
    assert not at.exception
    slider = at.select_slider[0]
    assert slider.options == ["0.1%", "0.5%", "1%", "3%", "5%", "10%", "20%"]
    assert len(set(slider.options)) == len(slider.options)  # 같은 라벨이 두 번 나오지 않음
    assert slider.value == 0.01  # 기본은 1%


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
    at = run("risk_monitor.py", local_ok=True)
    assert not at.exception
    assert at.metric[0].value == "10대"  # 1000대의 기본 1%
    assert len(at.dataframe) == 1


@pytest.mark.skipif(not REAL_ARTIFACT.exists(), reason="로컬 산출물이 있는 PC 에서만 확인 가능")
@pytest.mark.parametrize("rate", common.TICKS)
def test_risk_monitor_matches_overview_at_every_tick(rate):
    # 1번에서 정한 물량을 2번으로 넘기므로(화면 시안 B1) 눈금 일곱 곳 모두에서
    # 페이지 1(집계표)과 페이지 2(제품 단위)의 검사 대수·포착률이 같아야 함.
    # 2026-10-30 전에는 1% 한 곳만 확인했음
    overview = run("overview.py")
    overview.select_slider[0].set_value(rate).run()
    monitor = run("risk_monitor.py", local_ok=True, handoff_rate=rate)
    assert not overview.exception and not monitor.exception
    assert overview.metric[2].value == monitor.metric[0].value  # 검사 대수
    assert overview.metric[3].value == monitor.metric[2].value  # 포착률


def test_start_page_shows_headline():
    # 주소 / 로 들어온 사람이 처음 보는 화면. 대표 메시지가 1번과 같은 숫자여야 함
    at = run("start.py")
    assert not at.exception
    assert "11.5%" in at.title[0].value
    assert any("11.5배" in (m.proto.delta or "") for m in at.metric)


def test_missing_table_offers_build_guide(tmp_path, monkeypatch):
    # 집계표가 없으면 빠진 파일마다 오류 띠 + 만드는 법 버튼만 두고 나머지는 그리지 않음
    monkeypatch.setattr(common.figures, "REPORTS", tmp_path)
    at = run("overview.py")
    assert not at.exception
    assert len(at.error) == 3
    assert any(b.label == "집계표 만드는 법 보기" for b in at.button)
    assert len(at.select_slider) == 0


def test_risk_monitor_asks_before_showing_parts(tmp_path, monkeypatch):
    # 캡처 금지 확인 전에는 제품 값(KPI·목록)을 하나도 그리지 않음
    path = tmp_path / "parts.parquet"
    fake_parts(path)
    monkeypatch.setattr(common, "ARTIFACT", path)
    at = run("risk_monitor.py")
    assert not at.exception
    assert len(at.dataframe) == 0
    assert len(at.metric) == 0


def test_risk_monitor_broken_file_shows_guide(tmp_path, monkeypatch):
    # 깨진 파일이면 traceback 대신 오류 띠와 안내를 띄움 (화면 시안 D6)
    path = tmp_path / "parts.parquet"
    path.write_bytes(b"not a parquet file")
    monkeypatch.setattr(common, "ARTIFACT", path)
    at = run("risk_monitor.py", local_ok=True)
    assert not at.exception
    assert any("읽을 수 없음" in e.value for e in at.error)
    assert len(at.dataframe) == 0


def test_risk_monitor_takes_rate_from_overview(tmp_path, monkeypatch):
    # 1번에서 넘긴 물량이 2번 슬라이더의 시작값이 됨 (화면 시안 B1)
    path = tmp_path / "parts.parquet"
    fake_parts(path)
    monkeypatch.setattr(common, "ARTIFACT", path)
    at = run("risk_monitor.py", local_ok=True, handoff_rate=0.05)
    assert not at.exception
    assert at.select_slider[0].value == 0.05
    assert at.metric[0].value == "50대"


def test_risk_monitor_keeps_rate_after_leaving(tmp_path, monkeypatch):
    # 다른 페이지에 다녀와도 마지막 물량이 남아야 함. Streamlit 은 화면에 없던 위젯 값을
    # 지우므로 persist_state 로 남김 (2026-10-30 브라우저 확인에서 1% 로 돌아가던 결함)
    path = tmp_path / "parts.parquet"
    fake_parts(path)
    monkeypatch.setattr(common, "ARTIFACT", path)
    at = run("risk_monitor.py", local_ok=True)
    at.select_slider[0].set_value(0.05).run()
    at.switch_page("views/overview.py").run()
    at.switch_page("views/risk_monitor.py").run()
    assert not at.exception
    assert at.select_slider[0].value == 0.05


def test_risk_monitor_slider_moves_after_start(tmp_path, monkeypatch):
    # 넘겨받은 물량으로 시작한 뒤에도 사용자가 옮긴 값이 그대로 먹어야 함
    # (시작값을 매번 덮어쓰면 슬라이더가 되돌아감)
    path = tmp_path / "parts.parquet"
    fake_parts(path)
    monkeypatch.setattr(common, "ARTIFACT", path)
    at = run("risk_monitor.py", local_ok=True, handoff_rate=0.05)
    at.select_slider[0].set_value(0.10).run()
    assert not at.exception
    assert at.metric[0].value == "100대"


@pytest.mark.parametrize("page", ["start.py", "overview.py", "risk_monitor.py", "diagnostics.py"])
def test_every_page_shows_banner(page, tmp_path, monkeypatch):
    # 명세 0절 4번: 모든 페이지 제목 아래 안내 문구. 2번은 파일이 없는 공개 저장소 상태로 봄
    monkeypatch.setattr(common, "ARTIFACT", tmp_path / "없음.parquet")
    at = run(page)
    assert not at.exception
    assert any(c.value == common.BANNER for c in at.caption)


def test_start_page_without_tables_keeps_banner(tmp_path, monkeypatch):
    # 집계표가 없어도 제목·안내 문구가 먼저 나오고, 그 아래 오류 띠가 쌓임
    monkeypatch.setattr(common.figures, "REPORTS", tmp_path)
    at = run("start.py")
    assert not at.exception
    assert any(c.value == common.BANNER for c in at.caption)
    assert len(at.error) == 2


def test_risk_monitor_keeps_warning_before_and_after_confirm(tmp_path, monkeypatch):
    # 캡처·공유 금지 경고는 확인 전에도 후에도 머리에 남아야 함 (명세 0절 4번)
    path = tmp_path / "parts.parquet"
    fake_parts(path)
    monkeypatch.setattr(common, "ARTIFACT", path)
    for state in ({}, {"local_ok": True}):
        at = run("risk_monitor.py", **state)
        assert not at.exception
        assert any("캡처·공유하지 말 것" in w.value for w in at.warning)


def test_risk_monitor_refuses_without_local_config(tmp_path, monkeypatch):
    # 설정 파일이 안 읽혀 모든 주소에서 접속을 받는 상태면 제품 값을 그리지 않음
    path = tmp_path / "parts.parquet"
    fake_parts(path)
    monkeypatch.setattr(common, "ARTIFACT", path)
    st_config.set_option("server.address", "0.0.0.0")
    at = run("risk_monitor.py", local_ok=True)
    assert not at.exception
    assert len(at.dataframe) == 0
    assert any("config.toml" in e.value for e in at.error)


def test_config_keeps_page2_local():
    # 2번 목록은 이 PC 에서만, 내려받기 없이. 설정이 main.py 옆에 있어야 실행 폴더와 상관없이 읽힘
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["server"]["address"] == "localhost"
    assert cfg["client"]["disableDataExport"] is True
    assert cfg["theme"]["primaryColor"].lower() != "#ff4b4b"  # 기본 빨강은 '나쁜 결과' 색과 겹침


def test_overview_ticks_match_monitor_ticks():
    # 1번이 넘기는 값(곡선 표의 실수값을 반올림)이 2번 눈금에 그대로 있어야 함
    curve = common.table("T_포착률곡선")
    rates = curve.loc[curve["모델"] == "LightGBM", "목표_검사율"]
    assert sorted({round(r, 5) for r in rates if round(r, 5) in common.TICKS}) == list(common.TICKS)


def test_build_steps_follow_readme_order():
    # 집계표 만드는 순서가 화면과 README 두 곳에 있음. 한쪽만 고치면 어긋나므로 순서를 대조함
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    script = r"(?:src|tools)/\w+\.py"
    shown = re.findall(script, common.BUILD_STEPS)
    # README 1-2절의 실행 명령 줄(.venv/Scripts/python.exe ...)만 순서대로 모음
    readme_order = re.findall(r"^\.venv/Scripts/python\.exe (?:-W ignore )?(" + script + ")", readme, re.M)
    assert len(shown) == 10
    assert shown == [m for m in readme_order if m in shown]
