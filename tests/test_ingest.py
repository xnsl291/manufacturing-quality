####################################################################
# test_ingest.py — ingest.py 를 실제 Bosch 데이터 없이 검증함
#
# 왜 이런 테스트인가
#   Kaggle 원본은 규정 동의와 API 토큰이 있어야 받을 수 있고 그것은
#   사용자만 할 수 있음. 그래서 Bosch와 같은 모양(컬럼명 규칙, 결측
#   비율, 낮은 불량률)의 작은 합성 CSV를 만들어 변환 로직만 먼저
#   검증함.
#
#   특히 노린 것은 "조각마다 dtype이 흔들려 parquet 쓰기가 깨지는"
#   경우임. 어떤 컬럼이 앞 조각에서는 전부 결측이고 뒤 조각에서는
#   값이 있으면, 조각별로 스키마가 달라져 중간에 터지거나 조용히
#   값이 바뀜. Bosch는 결측이 80% 넘어서 실제로 잘 일어나는 일임.
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest import convert, count_csv_rows, parse_column  # noqa: E402


####################################################################
# 컬럼명 파싱
####################################################################
def test_parse_column_측정값():
    got = parse_column("L3_S36_F3938")
    assert got == {
        "column": "L3_S36_F3938",
        "line": 3,
        "station": 36,
        "kind": "F",
        "index": 3938,
        "station_key": "L3_S36",
    }


def test_parse_column_날짜():
    got = parse_column("L0_S0_D1")
    assert got is not None
    assert got["kind"] == "D"
    assert got["station_key"] == "L0_S0"


@pytest.mark.parametrize("name", ["Id", "Response", "L3_S36", "L3_S36_X1", "", "foo"])
def test_parse_column_규칙에_안_맞으면_None(name):
    assert parse_column(name) is None


####################################################################
# 행 수 세기
####################################################################
def test_count_csv_rows_끝에_줄바꿈_있는_경우(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"Id,x\n1,2\n3,4\n5,6\n")
    assert count_csv_rows(p) == 3


def test_count_csv_rows_끝에_줄바꿈_없는_경우(tmp_path):
    p = tmp_path / "b.csv"
    p.write_bytes(b"Id,x\n1,2\n3,4\n5,6")
    assert count_csv_rows(p) == 3


def test_count_csv_rows_버퍼_경계를_걸쳐도_맞음(tmp_path):
    p = tmp_path / "c.csv"
    p.write_bytes(b"Id,x\n" + b"1,2\n" * 5000)
    # 버퍼를 아주 작게 줘서 조각 경계를 여러 번 넘게 만든다
    assert count_csv_rows(p, buffer_size=7) == 5000


####################################################################
# 합성 Bosch 데이터
####################################################################
def make_bosch_like(path: Path, n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Bosch와 같은 모양의 CSV를 만든다.

    - 컬럼명이 L{라인}_S{스테이션}_F{번호} / _D{번호} 규칙을 따름
    - 결측이 대부분임 (제품이 일부 공정만 지나감)
    - 불량률이 1% 아래로 낮음
    - L0_S1_F2 는 **앞쪽 절반이 전부 결측**이 되게 만들어, 조각별
      스키마가 흔들리는 상황을 일부러 재현함
    """
    rng = np.random.default_rng(seed)
    cols = {
        "Id": np.arange(1, n_rows + 1, dtype=np.int64),
        "L0_S0_F0": rng.normal(size=n_rows),
        "L0_S1_F2": rng.normal(size=n_rows),
        "L3_S36_F3938": rng.normal(size=n_rows),
        "L0_S0_D1": rng.uniform(0, 1700, size=n_rows),
        "Response": (rng.random(n_rows) < 0.006).astype(np.int64),
    }
    df = pd.DataFrame(cols)

    # 공정을 안 지난 제품은 값이 없음 → 결측 70%
    for c in ["L0_S0_F0", "L3_S36_F3938", "L0_S0_D1"]:
        df.loc[rng.random(n_rows) < 0.70, c] = np.nan

    # 앞쪽 절반은 이 공정이 아예 없었던 것으로 둠
    df.loc[: n_rows // 2 - 1, "L0_S1_F2"] = np.nan

    df.to_csv(path, index=False)
    return df


def test_convert_행수와_컬럼수가_보존됨(tmp_path):
    src = tmp_path / "train_numeric.csv"
    original = make_bosch_like(src, n_rows=2_500)
    dst = tmp_path / "out.parquet"

    # 조각 크기를 행 수의 약수가 아니게 줘서 마지막 조각이 짧게 나오게 함
    written, n_cols, _ = convert(src, dst, chunksize=700)

    assert written == len(original) == count_csv_rows(src)
    assert n_cols == original.shape[1]

    back = pd.read_parquet(dst)
    assert len(back) == len(original)
    assert list(back.columns) == list(original.columns)


def test_convert_dtype가_줄어듦(tmp_path):
    src = tmp_path / "train_numeric.csv"
    make_bosch_like(src, n_rows=1_200)
    dst = tmp_path / "out.parquet"
    convert(src, dst, chunksize=500)

    back = pd.read_parquet(dst)
    assert back["Id"].dtype == np.int32
    assert back["Response"].dtype == np.int8
    for c in ["L0_S0_F0", "L0_S1_F2", "L3_S36_F3938", "L0_S0_D1"]:
        assert back[c].dtype == np.float32, c


def test_convert_전부_결측인_조각이_있어도_깨지지_않음(tmp_path):
    """이 테스트가 이 파일의 핵심임.

    L0_S1_F2 는 앞쪽 절반이 전부 결측임. 조각 크기를 그 절반보다
    작게 주면 "이 컬럼이 전부 결측인 조각"이 실제로 생김. dtype을
    미리 못 박지 않으면 이 지점에서 스키마가 흔들림.
    """
    src = tmp_path / "train_numeric.csv"
    original = make_bosch_like(src, n_rows=2_000)
    dst = tmp_path / "out.parquet"

    written, _, _ = convert(src, dst, chunksize=300)  # 1000행의 결측 구간을 여러 조각으로 쪼갬

    assert written == 2_000
    back = pd.read_parquet(dst)
    assert back["L0_S1_F2"].dtype == np.float32
    # 앞쪽 절반은 결측, 뒤쪽 절반은 값이 있어야 함
    assert back["L0_S1_F2"].iloc[:1000].isna().all()
    assert back["L0_S1_F2"].iloc[1000:].notna().all()
    assert int(back["Response"].sum()) == int(original["Response"].sum())


def test_convert_값이_보존됨(tmp_path):
    src = tmp_path / "train_numeric.csv"
    original = make_bosch_like(src, n_rows=900)
    dst = tmp_path / "out.parquet"
    convert(src, dst, chunksize=250)

    back = pd.read_parquet(dst)
    pd.testing.assert_series_equal(
        back["Id"].astype("int64"), original["Id"], check_dtype=False
    )
    # float32로 줄였으므로 float64 원본과는 오차 범위로 비교함
    np.testing.assert_allclose(
        back["L3_S36_F3938"].to_numpy(dtype="float64"),
        original["L3_S36_F3938"].to_numpy(dtype="float64"),
        rtol=1e-6,
        equal_nan=True,
    )


####################################################################
# 작업 메모
#   - 2026-09-30 작성. 실제 Bosch 데이터로는 아직 검증 못 했음.
#     여기서 통과한 것은 "합성 데이터에서 변환 로직이 맞다"는 뜻일 뿐,
#     "Bosch 원본이 잘 변환된다"는 뜻이 아님. 원본을 받은 뒤 다시 확인함
####################################################################
