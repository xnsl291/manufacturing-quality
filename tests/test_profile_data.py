####################################################################
# test_profile_data.py — Week 1 분석을 실제 Bosch 데이터 없이 검증함
#
# 검증 방식
#   정답을 내가 정해 놓고 합성 데이터를 만든 뒤, 계산이 그 정답을
#   되찾는지 본다. 예를 들어 "제품 3번은 L0_S1 공정을 지났지만 측정은
#   안 했다"를 일부러 만들어 넣고, 결측=미방문 검증이 그 칸을
#   '숫자만 결측'으로 세는지 확인한다.
#
#   통계적 성질이 아니라 **정해 둔 값의 복원**을 보는 테스트다. 실제
#   데이터가 어떻게 생겼는지는 모르기 때문에, 알 수 있는 것은 계산이
#   스스로 모순되지 않는지까지다.
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.profile_data import (  # noqa: E402
    build_column_map,
    check_missing_means_unvisited,
    leakage_candidates,
    path_table,
    split_feasibility,
    station_all_missing,
    time_structure,
)

# 합성 공장 — 공정 3개, 공정마다 측정값 개수가 다름(집계 교란을 보려면 달라야 함)
NUMERIC_COLS = {
    "L0_S0": ["L0_S0_F0", "L0_S0_F2"],
    "L0_S1": ["L0_S1_F4", "L0_S1_F6", "L0_S1_F8"],
    "L3_S36": ["L3_S36_F3938"],
}
DATE_COLS = {
    "L0_S0": ["L0_S0_D1"],
    "L0_S1": ["L0_S1_D5"],
    "L3_S36": ["L3_S36_D3939"],
}


def write_pair(
    tmp_path: Path,
    visit: np.ndarray,
    response: np.ndarray,
    *,
    numeric_only_missing: list[tuple[int, str]] | None = None,
    date_only_missing: list[tuple[int, str]] | None = None,
    start_offsets: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """방문 행렬을 받아 숫자/날짜 parquet 두 개를 짝이 맞게 만든다.

    visit  (제품수 × 3) bool. True면 그 공정을 지났음
    numeric_only_missing  (행, 공정) — 지났지만 측정 안 한 칸
    date_only_missing     (행, 공정) — 측정값은 있는데 시각이 없는 칸
    """
    stations = list(NUMERIC_COLS)
    n = len(response)
    rng = np.random.default_rng(7)

    num = {"Id": np.arange(1, n + 1, dtype=np.int64)}
    dat = {"Id": np.arange(1, n + 1, dtype=np.int64)}

    if start_offsets is None:
        start_offsets = np.arange(n, dtype=float) * 10.0

    for j, st in enumerate(stations):
        present = visit[:, j]
        for c in NUMERIC_COLS[st]:
            v = rng.normal(size=n)
            v[~present] = np.nan
            num[c] = v
        for k, c in enumerate(DATE_COLS[st]):
            v = start_offsets + j * 3.0 + k
            v[~present] = np.nan
            dat[c] = v

    for row, st in numeric_only_missing or []:
        for c in NUMERIC_COLS[st]:
            num[c][row] = np.nan
    for row, st in date_only_missing or []:
        for c in DATE_COLS[st]:
            dat[c][row] = np.nan

    num["Response"] = response.astype(np.int64)

    npath = tmp_path / "numeric.parquet"
    dpath = tmp_path / "date.parquet"
    pd.DataFrame(num).to_parquet(npath, index=False)
    pd.DataFrame(dat).to_parquet(dpath, index=False)
    return npath, dpath


####################################################################
# 컬럼 지도
####################################################################
def test_build_column_map_파싱과_규칙밖_이름(tmp_path):
    visit = np.ones((4, 3), dtype=bool)
    npath, _ = write_pair(tmp_path, visit, np.zeros(4, dtype=int))

    cmap, unparsed = build_column_map(npath)

    assert sorted(unparsed) == ["Id", "Response"]
    assert len(cmap) == 6  # 2 + 3 + 1
    assert set(cmap["station_key"]) == {"L0_S0", "L0_S1", "L3_S36"}
    counts = cmap.groupby("station_key").size().to_dict()
    assert counts == {"L0_S0": 2, "L0_S1": 3, "L3_S36": 1}


####################################################################
# 공정별 전부 결측 판정
####################################################################
def test_station_all_missing_방문행렬을_되찾음(tmp_path):
    visit = np.array(
        [
            [True, True, True],
            [True, False, False],
            [False, False, True],
            [False, False, False],
        ]
    )
    npath, _ = write_pair(tmp_path, visit, np.zeros(4, dtype=int))
    cmap, _ = build_column_map(npath)

    missing = station_all_missing(npath, cmap)

    # 열 순서는 station_key 정렬 순서
    assert list(missing.columns) == ["L0_S0", "L0_S1", "L3_S36"]
    np.testing.assert_array_equal(missing.to_numpy(), ~visit)


def test_station_all_missing_일부만_비면_미방문이_아님(tmp_path):
    """한 공정 안에서 측정값 일부만 비는 경우.

    이 테스트가 없으면 판정을 `all`에서 `any`로 바꿔도 아무 테스트가
    죽지 않음(2026-10-01 변형 검사에서 실제로 통과해 버렸음). 합성
    데이터가 공정 컬럼을 늘 한꺼번에 비우기 때문임.

    실제 공정에서는 측정 항목 몇 개만 누락될 수 있고, 그때 `any`를
    쓰면 그 제품이 공정을 지나지 않은 것으로 잘못 세게 됨.
    """
    visit = np.ones((3, 3), dtype=bool)
    npath, _ = write_pair(tmp_path, visit, np.zeros(3, dtype=int))

    # L0_S1 은 측정값 3개짜리임. 제품 1번에서 그중 하나만 비운다
    df = pd.read_parquet(npath)
    df.loc[1, "L0_S1_F4"] = np.nan
    df.to_parquet(npath, index=False)

    cmap, _ = build_column_map(npath)
    missing = station_all_missing(npath, cmap)

    # 일부만 비었으므로 "전부 결측"이 아니다 → 방문한 것으로 봐야 함
    assert not missing.loc[1, "L0_S1"]
    assert not missing["L0_S1"].any()


####################################################################
# 결측 = 미방문 가설 — 이 파일의 핵심
####################################################################
def test_결측미방문_완전히_일치하면_일치율_1(tmp_path):
    visit = np.array([[True, True, False], [False, True, True], [True, False, True]])
    npath, dpath = write_pair(tmp_path, visit, np.zeros(3, dtype=int))
    nmap, _ = build_column_map(npath)
    dmap, _ = build_column_map(dpath)

    summary, per_station = check_missing_means_unvisited(
        station_all_missing(npath, nmap), station_all_missing(dpath, dmap)
    )

    assert summary["전체 일치율"] == 1.0
    assert summary["숫자만 결측 칸"] == 0
    assert summary["날짜만 결측 칸"] == 0
    assert (per_station["일치율"] == 1.0).all()


def test_결측미방문_지났지만_측정안한_칸을_숫자만결측으로_셈(tmp_path):
    visit = np.ones((5, 3), dtype=bool)
    # 제품 1·3이 L0_S1을 지났지만 측정은 안 함 → 숫자만 결측 2칸
    npath, dpath = write_pair(
        tmp_path,
        visit,
        np.zeros(5, dtype=int),
        numeric_only_missing=[(1, "L0_S1"), (3, "L0_S1")],
    )
    nmap, _ = build_column_map(npath)
    dmap, _ = build_column_map(dpath)

    summary, per_station = check_missing_means_unvisited(
        station_all_missing(npath, nmap), station_all_missing(dpath, dmap)
    )

    assert summary["숫자만 결측 칸"] == 2
    assert summary["날짜만 결측 칸"] == 0
    row = per_station.set_index("station_key").loc["L0_S1"]
    assert row["숫자만 결측"] == 2
    assert row["방문·측정(양쪽 있음)"] == 3
    assert row["일치율"] == pytest.approx(3 / 5)


def test_결측미방문_시각만_없는_칸을_날짜만결측으로_셈(tmp_path):
    visit = np.ones((4, 3), dtype=bool)
    npath, dpath = write_pair(
        tmp_path,
        visit,
        np.zeros(4, dtype=int),
        date_only_missing=[(0, "L3_S36")],
    )
    nmap, _ = build_column_map(npath)
    dmap, _ = build_column_map(dpath)

    summary, _ = check_missing_means_unvisited(
        station_all_missing(npath, nmap), station_all_missing(dpath, dmap)
    )

    assert summary["날짜만 결측 칸"] == 1
    assert summary["숫자만 결측 칸"] == 0


def test_결측미방문_한쪽에만_있는_공정은_비교하지_않고_보고함():
    num = pd.DataFrame({"L0_S0": [True, False], "L0_S9": [True, True]})
    dat = pd.DataFrame({"L0_S0": [True, False], "L3_S36": [False, False]})

    summary, per_station = check_missing_means_unvisited(num, dat)

    assert summary["비교한 공정 수"] == 1
    assert summary["숫자에만 있는 공정"] == ["L0_S9"]
    assert summary["날짜에만 있는 공정"] == ["L3_S36"]
    assert list(per_station["station_key"]) == ["L0_S0"]


####################################################################
# 경로
####################################################################
def test_path_table_경로별_건수와_불량률(tmp_path):
    # 경로 A(세 공정 전부) 4개 중 불량 1개, 경로 B(첫 공정만) 2개 중 불량 2개
    visit = np.array(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, False, False],
            [True, False, False],
        ]
    )
    response = np.array([1, 0, 0, 0, 1, 1])
    npath, _ = write_pair(tmp_path, visit, response)
    cmap, _ = build_column_map(npath)
    missing = station_all_missing(npath, cmap)

    out = path_table(missing, pd.Series(response))

    assert len(out) == 2
    a = out[out["공정수"] == 3].iloc[0]
    b = out[out["공정수"] == 1].iloc[0]
    assert a["제품수"] == 4 and a["불량수"] == 1 and a["불량률"] == pytest.approx(0.25)
    assert b["제품수"] == 2 and b["불량수"] == 2 and b["불량률"] == pytest.approx(1.0)
    assert a["path"] == "L0_S0|L0_S1|L3_S36"
    assert b["path"] == "L0_S0"


def test_path_table_공정이_63개를_넘으면_멈춤():
    missing = pd.DataFrame(
        np.zeros((2, 64), dtype=bool), columns=[f"L0_S{i}" for i in range(64)]
    )
    with pytest.raises(ValueError, match="int64"):
        path_table(missing, pd.Series([0, 0]))


####################################################################
# 시간 구조
####################################################################
def test_time_structure_시작_종료_소요시간(tmp_path):
    visit = np.ones((3, 3), dtype=bool)
    offsets = np.array([0.0, 100.0, 200.0])
    _, dpath = write_pair(
        tmp_path, visit, np.zeros(3, dtype=int), start_offsets=offsets
    )
    dmap, _ = build_column_map(dpath)

    per_part, order = time_structure(dpath, dmap)

    # 공정 j의 날짜 = offset + j*3 → 시작은 offset, 종료는 offset+6
    np.testing.assert_allclose(per_part["start"].to_numpy(), offsets)
    np.testing.assert_allclose(per_part["end"].to_numpy(), offsets + 6.0)
    np.testing.assert_allclose(per_part["duration"].to_numpy(), [6.0, 6.0, 6.0])


def test_time_structure_공정순서를_날짜로_되찾음(tmp_path):
    visit = np.ones((5, 3), dtype=bool)
    _, dpath = write_pair(tmp_path, visit, np.zeros(5, dtype=int))
    dmap, _ = build_column_map(dpath)

    _, order = time_structure(dpath, dmap)

    # 합성 데이터에서 날짜는 공정 순서대로 커지게 만들었음
    assert list(order["station_key"]) == ["L0_S0", "L0_S1", "L3_S36"]
    assert list(order["추정_공정순서"]) == [1, 2, 3]


####################################################################
# 누설 후보
####################################################################
def test_leakage_상관이_없으면_배수가_1에_가까움():
    rng = np.random.default_rng(0)
    n = 20_000
    ids = pd.Series(np.arange(1, n + 1))
    start = pd.Series(np.arange(n, dtype=float))
    response = pd.Series((rng.random(n) < 0.05).astype(int))

    got = leakage_candidates(ids, start, response)

    assert got["Id-시작시각 순위상관"] == pytest.approx(1.0)
    assert got["배수"] == pytest.approx(1.0, abs=0.15)


def test_leakage_불량이_몰려_나오면_배수가_크게_나옴():
    """불량을 3개씩 묶음으로 심어 인접 상관을 만든다.

    주의 — 처음에는 "앞이 불량이면 이번도 불량"을 누적시켜 만들었는데,
    그러면 한 번 불량이 나온 뒤 끝까지 불량이 되어 전체 불량률까지
    1에 가까워지고 배수가 1이 되어 버렸음. 누설을 재현하려면 **전체
    불량률은 낮게 유지하면서 뭉치기만** 해야 함.

    불량 묶음 길이가 3이면, 앞이 불량인 행 중 불량인 비율은 2/3이고
    전체 불량률은 5%이므로 배수는 13 안팎이 되어야 함.
    """
    n = 30_000
    run_len = 3
    rng = np.random.default_rng(1)
    # 간격을 6 이상으로 띄워 묶음이 서로 붙지 않게 한다
    starts = rng.choice(np.arange(0, n - run_len, 6), size=n // 60, replace=False)

    response = np.zeros(n, dtype=int)
    for s in starts:
        response[s : s + run_len] = 1

    got = leakage_candidates(
        pd.Series(np.arange(1, n + 1)),
        pd.Series(np.arange(n, dtype=float)),
        pd.Series(response),
    )

    assert got["전체 불량률"] == pytest.approx(0.05, abs=0.005)
    assert got["앞 제품이 불량일 때 불량률"] == pytest.approx(2 / 3, abs=0.02)
    assert got["배수"] > 10.0


def test_leakage_Id가_시간과_거꾸로면_상관이_음수():
    n = 1_000
    got = leakage_candidates(
        pd.Series(np.arange(1, n + 1)),
        pd.Series(np.arange(n, 0, -1, dtype=float)),
        pd.Series(np.zeros(n, dtype=int)),
    )
    assert got["Id-시작시각 순위상관"] == pytest.approx(-1.0)


####################################################################
# 분할 가능성
####################################################################
def test_split_feasibility_구간이_같은_크기로_나뉨():
    n = 1_000
    out = split_feasibility(
        pd.Series(np.arange(n, dtype=float)), pd.Series(np.zeros(n, dtype=int)), n_bins=5
    )
    assert list(out["구간"]) == [1, 2, 3, 4, 5]
    assert set(out["제품수"]) == {200}


def test_split_feasibility_뒤_구간_불량률_상승을_잡아냄():
    n = 1_000
    response = np.zeros(n, dtype=int)
    response[800:] = 1  # 마지막 구간만 전부 불량
    out = split_feasibility(
        pd.Series(np.arange(n, dtype=float)), pd.Series(response), n_bins=5
    )
    rates = out.set_index("구간")["불량률"]
    assert rates.loc[1] == 0.0
    assert rates.loc[5] == pytest.approx(1.0)


def test_split_feasibility_시간순이_아닌_입력도_정렬해_처리함():
    n = 100
    order = np.random.default_rng(3).permutation(n)
    start = pd.Series(order.astype(float))
    response = pd.Series((order >= 80).astype(int))  # 시간상 뒤쪽이 불량

    out = split_feasibility(start, response, n_bins=5)

    rates = out.set_index("구간")["불량률"]
    assert rates.loc[1] == 0.0
    assert rates.loc[5] == pytest.approx(1.0)


####################################################################
# 작업 메모
#   - 2026-10-01 작성. 여기서 통과한 것은 "정해 둔 정답을 계산이
#     되찾는다"는 뜻일 뿐, "Bosch 원본에서 맞다"는 뜻이 아님
####################################################################
