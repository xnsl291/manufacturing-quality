####################################################################
# test_error_analysis.py — 오탐·미탐 집계가 맞는지 검증
#
# 노리는 것
#   ① 네 집단 분류가 뒤바뀌지 않는지. TP 와 FN 을 섞으면 "놓친 불량이
#      이렇게 다르다"는 결론이 통째로 반대가 된다.
#   ② 놓친 불량이 한쪽에 몰려 있을 때 그것을 실제로 드러내는지.
#      드러내지 못하면 이 분석은 숫자만 채운 장식이 된다.
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.error_analysis import classify, group_profile, path_mix, period_mix  # noqa: E402


####################################################################
# ① 분류
####################################################################
def test_classify_네_집단을_정확히_나눔():
    # 점수 상위 2개(0.9, 0.8)가 알람. 정답은 [1, 0, 1, 0]
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1])
    g = classify(y, score, alert_rate=0.5)

    assert g.tolist() == ["TP_잡음", "FP_헛수고", "FN_놓침", "TN_정상"]


def test_classify_집단_크기가_검사율과_불량수에_맞음():
    rng = np.random.default_rng(0)
    n = 10_000
    y = (rng.random(n) < 0.01).astype(int)
    score = rng.random(n)
    g = classify(y, score, alert_rate=0.05)

    counts = g.value_counts()
    alerted = counts.get("TP_잡음", 0) + counts.get("FP_헛수고", 0)
    defects = counts.get("TP_잡음", 0) + counts.get("FN_놓침", 0)
    assert alerted == pytest.approx(n * 0.05, rel=0.05)
    assert defects == int(y.sum())


####################################################################
# ② 놓친 불량의 체계적 차이를 드러내는가
####################################################################
def test_group_profile_놓친_불량이_공정수가_적으면_드러남():
    """놓친 불량은 공정을 적게 지나고, 잡은 불량은 많이 지나게 만든다.
    집계표의 중앙값이 그 차이를 보여야 한다."""
    n = 400
    y = np.zeros(n, dtype=int)
    score = np.zeros(n)
    visited = np.full(n, 30)

    # 앞 20개: 불량이고 점수가 높다(TP). 공정 40개를 지남
    y[:20] = 1; score[:20] = 0.9; visited[:20] = 40
    # 다음 20개: 불량인데 점수가 낮다(FN). 공정 10개만 지남
    y[20:40] = 1; score[20:40] = 0.0; visited[20:40] = 10

    feats = pd.DataFrame(
        {
            "visited_station_count": visited,
            "visited_line_count": np.full(n, 3),
            "measurement_count": visited * 10,
            "missing_ratio": 1 - visited / 52,
            "duration": np.full(n, 5.0),
            "start_time": np.arange(n, dtype=float),
        }
    )
    g = classify(y, score, alert_rate=0.05)  # 상위 5% = 20개 → 정확히 TP 20개
    prof = group_profile(g, feats, score)

    assert prof.loc["TP_잡음", "제품수"] == 20
    assert prof.loc["FN_놓침", "제품수"] == 20
    assert prof.loc["TP_잡음", "visited_station_count"] == 40
    assert prof.loc["FN_놓침", "visited_station_count"] == 10
    # 놓친 불량이 결측 비율이 더 높아야 한다
    assert prof.loc["FN_놓침", "missing_ratio"] > prof.loc["TP_잡음", "missing_ratio"]


def test_group_profile_꼬리가_긴_값에서_중앙값을_씀():
    """공정시간은 꼬리가 길다. 실제 데이터에서 중앙 3.70 / 평균 10.72 /
    최대 699.20 이었다. 평균으로 집계하면 한두 건에 끌려가 "놓친 불량의
    공정시간이 길다"는 틀린 결론이 나올 수 있다.

    여기서는 놓친 불량 20건 중 **한 건만** 공정시간을 1000으로 둔다.
    중앙값은 5 로 그대로이고 평균은 55 로 뛴다. 집계표에 5 가 나와야 한다.
    """
    n = 400
    y = np.zeros(n, dtype=int); score = np.zeros(n)
    duration = np.full(n, 5.0)
    y[:20] = 1; score[:20] = 0.9                 # TP
    y[20:40] = 1; score[20:40] = 0.0             # FN
    duration[20] = 1000.0                        # FN 한 건만 극단값

    feats = pd.DataFrame(
        {
            "visited_station_count": np.full(n, 30),
            "visited_line_count": np.full(n, 3),
            "measurement_count": np.full(n, 300),
            "missing_ratio": np.full(n, 0.4),
            "duration": duration,
            "start_time": np.arange(n, dtype=float),
        }
    )
    prof = group_profile(classify(y, score, 0.05), feats, score)

    assert prof.loc["FN_놓침", "duration"] == pytest.approx(5.0), \
        "평균으로 집계하면 55 가 나온다 — 중앙값을 써야 함"


def test_group_profile_집단_순서가_고정됨():
    n = 100
    y = np.zeros(n, dtype=int); y[:10] = 1
    score = np.linspace(1, 0, n)
    feats = pd.DataFrame({c: np.ones(n) for c in
                          ["visited_station_count", "visited_line_count",
                           "measurement_count", "missing_ratio", "duration", "start_time"]})
    prof = group_profile(classify(y, score, 0.05), feats, score)
    assert list(prof.index)[:2] == ["TP_잡음", "FN_놓침"]


####################################################################
# 경로·시기 비중
####################################################################
def test_path_mix_놓친_불량이_특정_경로에_몰리면_드러남():
    n = 200
    y = np.zeros(n, dtype=int); score = np.zeros(n)
    path = np.zeros(n, dtype=int)

    y[:10] = 1; score[:10] = 0.9; path[:10] = 1    # TP 는 경로 1
    y[10:20] = 1; score[10:20] = 0.0; path[10:20] = 7  # FN 은 경로 7
    path[20:] = 1

    mix = path_mix(pd.Series(classify(y, score, 0.05)), pd.Series(path), top=3)
    assert mix.loc["TP_잡음", 1] == pytest.approx(1.0)
    assert mix.loc["FN_놓침", 7] == pytest.approx(1.0)


def test_period_mix_놓친_불량이_특정_시기에_몰리면_드러남():
    """입력 행 순서를 시작 시각과 **일부러 어긋나게** 섞는다.

    처음에는 start_time 을 그냥 arange 로 줬는데, 그러면 입력 순서가
    이미 시간순이라 내부 정렬을 빼도 결과가 같아서 테스트가 통과했음
    (2026-10-06 변형 검사에서 실제로 통과해 버렸음).
    """
    n = 500
    y = np.zeros(n, dtype=int); score = np.zeros(n)
    start = np.arange(n, dtype=float)

    y[400:425] = 1; score[400:425] = 0.0  # 놓친 불량 → 시간상 마지막 시기
    y[:25] = 1; score[:25] = 0.9          # 잡은 불량 → 시간상 첫 시기

    # 행 순서를 뒤섞는다. 시간 정보는 start 에만 남는다
    perm = np.random.default_rng(11).permutation(n)
    y, score, start = y[perm], score[perm], start[perm]

    mix = period_mix(pd.Series(classify(y, score, 0.05)), pd.Series(start), n_bins=5)
    assert mix.loc["FN_놓침", 5] == pytest.approx(1.0)
    assert mix.loc["TP_잡음", 1] == pytest.approx(1.0)


def test_period_mix_구간이_같은_크기로_나뉨():
    """주의 — 처음에는 점수를 한 개만 1.0 으로 두고 나머지를 전부 0 으로
    만들었는데, 전부 동점이라 분위수 임계값이 0 이 되어 **전부 알람**이
    되고 TN 집단이 아예 생기지 않았음(KeyError). 점수를 서로 다르게
    줘야 네 집단이 생긴다."""
    n = 1000
    y = np.zeros(n, dtype=int); y[0] = 1
    score = np.linspace(1.0, 0.0, n)  # 동점 없음
    mix = period_mix(pd.Series(classify(y, score, 0.1)), pd.Series(np.arange(n, dtype=float)))

    assert "TN_정상" in mix.index
    assert mix.loc["TN_정상"].sum() == pytest.approx(1.0)
    assert mix.shape[1] == 5
    # 알람은 점수 상위 10% = 앞 100개 → 시기 1에 몰려야 한다
    assert mix.loc["FP_헛수고", 1] == pytest.approx(1.0)


####################################################################
# 작업 메모
#   - 2026-10-06 작성. 개별 제품 표는 만들지 않으므로 테스트도 집계만 봄
####################################################################
