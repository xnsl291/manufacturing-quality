####################################################################
# test_early_prediction.py — 예측 시점을 앞당길 때 누설이 없는지 검증
#
# 노리는 것
#   시점 t 에서 자르면 **그 시점에 아직 모르는 값**이 입력에 들어가면
#   안 된다. 가장 쉽게 새는 것 셋이다.
#     ① 아직 안 지난 공정의 방문 여부·측정값
#     ② 총 공정시간(종료 시각) — 제품이 끝나야 알 수 있음
#     ③ 전체 방문 공정 수 — 같은 이유
#   이게 새면 조기예측 곡선이 통째로 거짓이 되고, "예측을 앞당기면
#   성능이 이만큼 떨어진다"는 이 프로젝트의 차별화 지점이 무너진다.
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.early_prediction import build_restricted, stations_up_to  # noqa: E402

# 합성 공장 — 공정 셋이 상대시각 0 / 5 / 20 에 있다.
# 실제 데이터처럼 **라인 번호가 공정 순서와 다르게** 만든다
# (L2_S9 가 가장 마지막이고 L3_S5 가 중간).
ORDER = pd.DataFrame(
    [
        {"station_key": "L0_S0", "상대시각_중앙값": 0.0, "순서_신뢰": "보통"},
        {"station_key": "L3_S5", "상대시각_중앙값": 5.0, "순서_신뢰": "보통"},
        {"station_key": "L2_S9", "상대시각_중앙값": 20.0, "순서_신뢰": "낮음(표본 적음)"},
    ]
)
STATIONS = ["L0_S0", "L3_S5", "L2_S9"]
OFFSETS = {"L0_S0": 0.0, "L3_S5": 5.0, "L2_S9": 20.0}


def write_pair(tmp_path: Path, visit: np.ndarray, n: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(4)
    base = np.arange(n, dtype=float) * 100.0
    num = {"Id": np.arange(1, n + 1, dtype=np.int64)}
    dat = {"Id": np.arange(1, n + 1, dtype=np.int64)}
    for j, st in enumerate(STATIONS):
        present = visit[:, j]
        line = int(st[1])
        v = base + OFFSETS[st]
        v[~present] = np.nan
        dat[f"L{line}_S{st.split('_S')[1]}_D{j}"] = v
        m = rng.normal(size=n)
        m[~present] = np.nan
        num[f"L{line}_S{st.split('_S')[1]}_F{j}"] = m
    num["Response"] = np.zeros(n, dtype=np.int64)
    npath, dpath = tmp_path / "numeric.parquet", tmp_path / "date.parquet"
    pd.DataFrame(num).to_parquet(npath, index=False)
    pd.DataFrame(dat).to_parquet(dpath, index=False)
    return npath, dpath


####################################################################
# 시점 정의
####################################################################
def test_stations_up_to_공정_진행도로_자름():
    early, info = stations_up_to(ORDER, 0.0)
    assert early == ["L0_S0"]
    assert info["포함 공정 수"] == 1 and info["전체 공정 수"] == 3

    mid, _ = stations_up_to(ORDER, 5.0)
    assert mid == ["L0_S0", "L3_S5"]

    allv, info = stations_up_to(ORDER, 20.0)
    assert allv == STATIONS
    assert info["순서 신뢰 낮은 공정"] == ["L2_S9"]


def test_라인번호가_아니라_진행도로_자르는지():
    """실제 데이터에서 L2 가 가장 마지막이었다. 라인 번호로 잘랐다면
    L0~L2 가 L3 를 빼고 L2 를 넣는 잘못이 된다. 진행도로 자르면
    중간 시점에 L3 가 들어가고 L2 는 빠져야 한다."""
    mid, _ = stations_up_to(ORDER, 5.0)
    assert "L3_S5" in mid, "중간 시점에 L3 가 들어가야 함"
    assert "L2_S9" not in mid, "가장 마지막 공정인 L2 는 빠져야 함"


####################################################################
# 누설 — 이 파일의 핵심
####################################################################
def test_자른_시점_이후_공정의_측정값이_입력에_없음(tmp_path):
    n = 40
    visit = np.ones((n, 3), dtype=bool)
    npath, dpath = write_pair(tmp_path, visit, n)

    allowed, _ = stations_up_to(ORDER, 0.0)  # L0_S0 만
    X = build_restricted(npath, dpath, allowed)

    cols = list(X.columns)
    assert any(c.startswith("L0_S0_F") for c in cols)
    assert not any(c.startswith("L3_S5_F") for c in cols), "아직 안 지난 공정의 측정값이 들어감"
    assert not any(c.startswith("L2_S9_F") for c in cols), "아직 안 지난 공정의 측정값이 들어감"
    assert "visited_L0_S0" in cols
    assert "visited_L3_S5" not in cols, "아직 안 지난 공정의 방문 여부가 들어감"


def test_총_공정시간과_종료시각이_입력에_없음(tmp_path):
    n = 30
    npath, dpath = write_pair(tmp_path, np.ones((n, 3), dtype=bool), n)
    allowed, _ = stations_up_to(ORDER, 5.0)
    X = build_restricted(npath, dpath, allowed)

    assert "duration" not in X.columns, "총 공정시간은 제품이 끝나야 알 수 있음"
    assert "end_time" not in X.columns, "종료 시각은 제품이 끝나야 알 수 있음"
    assert "start_time" in X.columns      # 시작 시각은 알 수 있음
    assert "elapsed_so_far" in X.columns  # 그 시점까지의 경과 시간은 알 수 있음


def test_경과시간이_자른_시점까지만_반영됨(tmp_path):
    """합성 데이터의 공정 날짜는 시작 + {0, 5, 20} 이다.
    L0_S0 까지만 자르면 경과 시간은 0, L3_S5 까지면 5 여야 한다.
    전체 기준(20)이 나오면 미래 정보가 들어간 것이다."""
    n = 25
    npath, dpath = write_pair(tmp_path, np.ones((n, 3), dtype=bool), n)

    X0 = build_restricted(npath, dpath, stations_up_to(ORDER, 0.0)[0])
    np.testing.assert_allclose(X0["elapsed_so_far"].to_numpy(), np.zeros(n), atol=1e-4)

    X1 = build_restricted(npath, dpath, stations_up_to(ORDER, 5.0)[0])
    np.testing.assert_allclose(X1["elapsed_so_far"].to_numpy(), np.full(n, 5.0), rtol=1e-4)

    X2 = build_restricted(npath, dpath, stations_up_to(ORDER, 20.0)[0])
    np.testing.assert_allclose(X2["elapsed_so_far"].to_numpy(), np.full(n, 20.0), rtol=1e-4)


def test_방문_공정수가_자른_시점까지만_셈(tmp_path):
    n = 10
    visit = np.ones((n, 3), dtype=bool)
    visit[:, 1] = False  # 가운데 공정은 아무도 안 지남
    npath, dpath = write_pair(tmp_path, visit, n)

    X_mid = build_restricted(npath, dpath, stations_up_to(ORDER, 5.0)[0])
    # 허용 공정 2개 중 1개만 방문
    assert set(X_mid["visited_station_count"]) == {1}

    X_all = build_restricted(npath, dpath, stations_up_to(ORDER, 20.0)[0])
    assert set(X_all["visited_station_count"]) == {2}


def test_측정값_개수도_자른_시점까지만_셈(tmp_path):
    n = 12
    npath, dpath = write_pair(tmp_path, np.ones((n, 3), dtype=bool), n)

    X0 = build_restricted(npath, dpath, stations_up_to(ORDER, 0.0)[0])
    assert set(X0["measurement_count"]) == {1}      # L0_S0 의 측정값 1개
    assert set(X0["missing_count"]) == {0}

    X2 = build_restricted(npath, dpath, stations_up_to(ORDER, 20.0)[0])
    assert set(X2["measurement_count"]) == {3}


####################################################################
# 작업 메모
#   - 2026-10-06 작성. 합성 공장의 라인 번호를 일부러 공정 순서와
#     어긋나게 만들었음(L2 가 마지막). 실제 데이터가 그랬기 때문
####################################################################
