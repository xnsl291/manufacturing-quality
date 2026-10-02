####################################################################
# test_features.py — 공정 파생 입력값과 시간 분할을 검증함
#
# 특히 노리는 것 두 가지
#   ① 방문 여부를 **날짜 쪽**에서 만드는지. 실제 데이터에 L3_S42,
#      L3_S46 처럼 날짜만 있고 측정값이 없는 공정이 있어서, 숫자 쪽만
#      보면 그 공정을 지난 사실을 잃는다.
#   ② 경로 상위 N개를 **학습 구간에서만** 고르는지. 전체에서 고르면
#      검증·시험 구간의 분포가 학습 설계에 새어 들어온다(누설).
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import (  # noqa: E402
    assign_split,
    build,
    column_map,
    measurement_counts,
    path_ids,
    station_time_matrix,
)

# 합성 공장 — L0_S1 은 날짜만 있고 측정값이 없는 공정(실제 L3_S42·L3_S46 역할)
NUMERIC_COLS = {"L0_S0": ["L0_S0_F0", "L0_S0_F2"], "L3_S36": ["L3_S36_F3938"]}
DATE_COLS = {"L0_S0": ["L0_S0_D1"], "L0_S1": ["L0_S1_D5"], "L3_S36": ["L3_S36_D3939"]}
DATE_STATIONS = list(DATE_COLS)  # L0_S0, L0_S1, L3_S36


def write_pair(tmp_path: Path, visit: np.ndarray, response: np.ndarray,
               start_offsets: np.ndarray | None = None) -> tuple[Path, Path]:
    """visit 은 (제품수 × 3) — 날짜 공정 3개 기준 방문 여부."""
    n = len(response)
    rng = np.random.default_rng(5)
    if start_offsets is None:
        start_offsets = np.arange(n, dtype=float) * 10.0

    num = {"Id": np.arange(1, n + 1, dtype=np.int64)}
    dat = {"Id": np.arange(1, n + 1, dtype=np.int64)}

    for j, st in enumerate(DATE_STATIONS):
        present = visit[:, j]
        for c in DATE_COLS[st]:
            v = start_offsets + j * 2.0
            v[~present] = np.nan
            dat[c] = v
        for c in NUMERIC_COLS.get(st, []):  # L0_S1 은 측정값 컬럼이 없다
            v = rng.normal(size=n)
            v[~present] = np.nan
            num[c] = v

    num["Response"] = response.astype(np.int64)
    npath, dpath = tmp_path / "numeric.parquet", tmp_path / "date.parquet"
    pd.DataFrame(num).to_parquet(npath, index=False)
    pd.DataFrame(dat).to_parquet(dpath, index=False)
    return npath, dpath


####################################################################
# ① 날짜 전용 공정
####################################################################
def test_방문여부가_날짜쪽_공정까지_모두_잡힘(tmp_path):
    visit = np.array([[True, True, True], [True, False, True], [True, True, False]])
    npath, dpath = write_pair(tmp_path, visit, np.zeros(3, dtype=int))

    nmap, dmap = column_map(npath), column_map(dpath)
    # 숫자 쪽에는 공정이 2개뿐이지만 날짜 쪽에는 3개다
    assert set(nmap["station_key"]) == {"L0_S0", "L3_S36"}
    assert set(dmap["station_key"]) == {"L0_S0", "L0_S1", "L3_S36"}

    visited, _ = station_time_matrix(dpath, dmap)
    assert list(visited.columns) == ["L0_S0", "L0_S1", "L3_S36"]
    np.testing.assert_array_equal(visited.to_numpy(), visit)


def test_build가_날짜전용_공정의_방문플래그를_만듦(tmp_path):
    visit = np.array([[True, True, True], [True, False, True]])
    npath, dpath = write_pair(tmp_path, visit, np.zeros(2, dtype=int))

    feats, info = build(npath, dpath)

    # 측정값이 없는 L0_S1 도 방문 플래그가 있어야 한다
    assert "visited_L0_S1" in feats.columns
    assert list(feats["visited_L0_S1"]) == [1, 0]
    assert info["방문 플래그 공정 수"] == 3
    assert list(feats["visited_station_count"]) == [3, 2]


####################################################################
# ② 경로 상위 N개를 학습 구간에서만 고르는가
####################################################################
def test_경로_상위N개를_학습구간에서만_고름():
    """전체 최다 경로와 학습 구간 최다 경로를 **다르게** 만들어야 의미가 있다.

    처음에는 학습 6개 / 시험 4개로 만들었는데, 그러면 전체에서 골라도
    학습에서 골라도 같은 경로가 1위라서 누설을 넣어도 테스트가 통과했음
    (2026-10-02 변형 검사에서 실제로 통과해 버렸음).

    그래서 학습 구간은 A가 3개뿐이고 시험 구간에 B가 7개 있게 만든다.
    전체 1위는 B, 학습 1위는 A다. 학습 구간에서 골라야 A가 1번이 된다.
    """
    A = [True, True, False]
    B = [False, True, True]
    visited = pd.DataFrame([A] * 3 + [B] * 7, columns=["s0", "s1", "s2"])
    split = pd.Series(["train"] * 3 + ["test"] * 7)

    out = path_ids(visited, split, top_n=1)

    assert list(out["path_id"][:3]) == [1] * 3  # 학습 구간 경로 A 가 1번
    assert list(out["path_id"][3:]) == [0] * 7  # 전체 1위였던 B 는 기타(0)


def test_경로키가_서로_다른_경로를_구분함():
    visited = pd.DataFrame(
        [[True, False, False], [False, True, False], [True, False, False]],
        columns=["s0", "s1", "s2"],
    )
    out = path_ids(visited, pd.Series(["train"] * 3), top_n=5)
    assert out["path_key"].iloc[0] == out["path_key"].iloc[2]
    assert out["path_key"].iloc[0] != out["path_key"].iloc[1]


def test_경로_공정이_63개_넘으면_멈춤():
    visited = pd.DataFrame(np.zeros((2, 64), dtype=bool), columns=[f"s{i}" for i in range(64)])
    with pytest.raises(ValueError, match="int64"):
        path_ids(visited, pd.Series(["train", "train"]))


####################################################################
# 시간 분할
####################################################################
def test_분할이_시작시각_순서대로_60_20_20():
    start = pd.Series(np.arange(100, dtype=float))
    split = assign_split(start)
    assert list(split[:60].unique()) == ["train"]
    assert list(split[60:80].unique()) == ["valid"]
    assert list(split[80:].unique()) == ["test"]


def test_분할이_입력_순서와_무관하게_시간순을_따름():
    rng = np.random.default_rng(1)
    order = rng.permutation(100)
    split = assign_split(pd.Series(order.astype(float)))
    # 시간상 앞 60개(값 0~59)가 train 이어야 한다
    assert set(split[order < 60]) == {"train"}
    assert set(split[(order >= 60) & (order < 80)]) == {"valid"}
    assert set(split[order >= 80]) == {"test"}


def test_날짜가_없는_제품은_학습구간으로_감():
    start = pd.Series([np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    split = assign_split(start)
    assert split.iloc[0] == "train" and split.iloc[1] == "train"
    # 뒤쪽 시각은 시험 구간에 들어가야 한다
    assert split.iloc[-1] == "test"


####################################################################
# 측정값 개수
####################################################################
def test_measurement_counts_개수와_결측비율(tmp_path):
    visit = np.array([[True, True, True], [True, False, False], [False, False, False]])
    npath, _ = write_pair(tmp_path, visit, np.zeros(3, dtype=int))
    nmap = column_map(npath)

    out = measurement_counts(npath, nmap)

    # 측정값 컬럼은 L0_S0 2개 + L3_S36 1개 = 3개
    assert list(out["measurement_count"]) == [3, 2, 0]
    assert list(out["missing_count"]) == [0, 1, 3]
    np.testing.assert_allclose(out["missing_ratio"].to_numpy(), [0.0, 1 / 3, 1.0], atol=1e-6)


####################################################################
# 전체 조립
####################################################################
def test_build_시간구조와_구간이_함께_나옴(tmp_path):
    n = 50
    visit = np.ones((n, 3), dtype=bool)
    offsets = np.arange(n, dtype=float) * 10.0
    npath, dpath = write_pair(tmp_path, visit, np.zeros(n, dtype=int), start_offsets=offsets)

    feats, info = build(npath, dpath)

    assert len(feats) == n
    np.testing.assert_allclose(feats["start_time"].to_numpy(), offsets, rtol=1e-5)
    # 공정 j 의 날짜 = 시작 + j*2 이므로 종료는 시작+4, 소요시간 4
    np.testing.assert_allclose(feats["duration"].to_numpy(), np.full(n, 4.0), rtol=1e-5)
    assert info["구간별 행 수"] == {"train": 30, "valid": 10, "test": 10}
    assert info["날짜가 없는 제품"] == 0
    assert feats["split"].tolist() == ["train"] * 30 + ["valid"] * 10 + ["test"] * 10


####################################################################
# 작업 메모
#   - 2026-10-02 작성. 합성 데이터의 L0_S1 이 실제 데이터의 L3_S42·L3_S46
#     (날짜만 있고 측정값 없는 공정) 역할을 함
####################################################################
