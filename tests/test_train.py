####################################################################
# test_train.py — 누설 입력값과 공정 요약 입력값을 검증함
#
# 특히 노리는 것
#   누설 입력값(`id_neighbor_features`)이 **원래 행 순서로 되돌아오는지**.
#   Id 로 정렬해 계산한 뒤 되돌리지 않으면 값이 엉뚱한 제품에 붙는다.
#   그러면 누설 실험 결과가 통째로 거짓이 되고, "누설이 실재한다"는
#   이 프로젝트의 차별화 지점이 무너진다.
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import column_map  # noqa: E402
from src.train import id_neighbor_features, station_summary  # noqa: E402


####################################################################
# 누설 입력값
####################################################################
def test_누설입력_Id가_이미_정렬된_경우():
    ids = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 1, 0, 0, 1])

    out = id_neighbor_features(ids, y)

    # 앞 제품의 정답: [nan, 0, 1, 0, 0]
    assert np.isnan(out["leak_prev_response"].iloc[0])
    np.testing.assert_array_equal(out["leak_prev_response"].to_numpy()[1:], [0, 1, 0, 0])
    # 뒤 제품의 정답: [1, 0, 0, 1, nan]
    np.testing.assert_array_equal(out["leak_next_response"].to_numpy()[:-1], [1, 0, 0, 1])
    assert np.isnan(out["leak_next_response"].iloc[-1])


def test_누설입력이_원래_행순서로_되돌아옴():
    """이 테스트가 이 파일의 핵심임.

    Id 가 뒤섞여 들어와도 결과는 **입력 행 순서**에 맞아야 한다.
    Id 순서로 계산한 뒤 되돌리지 않으면 값이 다른 제품에 붙는다.
    """
    # Id 순서는 3, 1, 5, 2, 4 로 뒤섞여 있다
    ids = np.array([3, 1, 5, 2, 4])
    y = np.array([1, 0, 1, 0, 0])
    # Id 순 정렬:  Id 1(y=0), 2(y=0), 3(y=1), 4(y=0), 5(y=1)
    # 따라서 Id 3 의 앞 제품은 Id 2 → y=0, 뒤 제품은 Id 4 → y=0
    #        Id 1 의 앞은 없음(nan), 뒤는 Id 2 → y=0
    #        Id 5 의 앞은 Id 4 → y=0, 뒤는 없음(nan)

    out = id_neighbor_features(ids, y)

    prev = out["leak_prev_response"].to_numpy()
    nxt = out["leak_next_response"].to_numpy()
    assert prev[0] == 0 and nxt[0] == 0          # 입력 0번 = Id 3
    assert np.isnan(prev[1]) and nxt[1] == 0     # 입력 1번 = Id 1
    assert prev[2] == 0 and np.isnan(nxt[2])     # 입력 2번 = Id 5
    assert prev[3] == 0 and nxt[3] == 1          # 입력 3번 = Id 2 (앞 Id1=0, 뒤 Id3=1)
    assert prev[4] == 1 and nxt[4] == 1          # 입력 4번 = Id 4 (앞 Id3=1, 뒤 Id5=1)


def test_누설입력_Id_간격이_맞음():
    ids = np.array([10, 40, 20])  # Id 순: 10, 20, 40
    y = np.array([0, 0, 0])
    out = id_neighbor_features(ids, y)
    gaps_next = out["leak_next_id_gap"].to_numpy()
    assert gaps_next[0] == 10   # Id 10 → 다음 20, 간격 10
    assert gaps_next[2] == 20   # Id 20 → 다음 40, 간격 20
    assert np.isnan(gaps_next[1])  # Id 40 은 마지막


def test_누설입력이_강한_신호여야_함():
    """불량이 뭉쳐 있으면 앞 제품의 정답이 강한 예측력을 가진다 —
    그래서 누설인 것이다. 신호가 없다면 실험 자체가 의미 없다."""
    n = 30_000
    rng = np.random.default_rng(2)
    y = np.zeros(n, dtype=int)
    for s in rng.choice(np.arange(0, n - 3, 6), size=n // 60, replace=False):
        y[s : s + 3] = 1
    ids = np.arange(1, n + 1)

    out = id_neighbor_features(ids, y)
    prev = out["leak_prev_response"].to_numpy()
    base = y.mean()
    cond = y[prev == 1].mean()
    assert cond / base > 5, f"누설 신호가 약함: {cond/base:.2f}배"


####################################################################
# 공정 요약 입력값
####################################################################
def test_station_summary_공정별_평균과_표준편차(tmp_path):
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "L0_S0_F0": [1.0, 10.0, np.nan],
            "L0_S0_F2": [3.0, 20.0, np.nan],
            "L3_S36_F3938": [5.0, np.nan, 7.0],
            "Response": [0, 0, 1],
        }
    )
    p = tmp_path / "numeric.parquet"
    df.to_parquet(p, index=False)

    out = station_summary(p, column_map(p))

    assert list(out.columns) == ["L0_S0_mean", "L0_S0_std", "L3_S36_mean", "L3_S36_std"]
    np.testing.assert_allclose(out["L0_S0_mean"].to_numpy(), [2.0, 15.0, np.nan])
    # 표본표준편차: [1,3] → 1.4142
    np.testing.assert_allclose(out["L0_S0_std"].to_numpy()[:2], [1.41421, 7.07107], rtol=1e-4)
    # 공정을 지나지 않은 제품은 결측으로 남아야 한다 (뒤에서 플래그를 만들 수 있게)
    assert np.isnan(out["L0_S0_mean"].iloc[2])


####################################################################
# 작업 메모
#   - 2026-10-02 작성. 누설 입력값은 실험 전용이고 최종 모델에 들어가지 않음
####################################################################
