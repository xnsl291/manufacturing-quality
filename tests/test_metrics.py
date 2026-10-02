####################################################################
# test_metrics.py — 이 프로젝트의 결론을 만드는 계산을 검증함
#
# 검증 방식
#   정답을 손으로 계산할 수 있는 작은 예를 만들어 비교한다. 지표는
#   포트폴리오의 헤드라인 숫자가 되므로, 한 자리 틀리면 결론이 틀린다.
#
# 특히 노리는 것
#   ① 완벽한 모델과 무작위 모델의 Lift 가 각각 이론값과 맞는가
#   ② Lift 를 **실제** 검사율로 나누는가 (목표로 나누면 거짓이 됨)
#   ③ 기간이 바뀌어 확률 분포가 밀렸을 때 고정 임계값의 검사율 이탈을
#      잡아내는가 — 실제 데이터에서 일어난 상황
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import (  # noqa: E402
    cost_curve,
    summary,
    threshold_at_alert_rate,
    threshold_stability,
    threshold_table,
)


####################################################################
# 임계값
####################################################################
def test_threshold_at_alert_rate_상위_10퍼센트():
    score = np.arange(100, dtype=float)  # 0..99
    thr = threshold_at_alert_rate(score, 0.10)
    assert (score >= thr).sum() == pytest.approx(10, abs=1)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_threshold_at_alert_rate_범위_밖이면_멈춤(bad):
    with pytest.raises(ValueError, match="alert_rate"):
        threshold_at_alert_rate(np.arange(10, dtype=float), bad)


####################################################################
# ① 완벽한 모델 / 무작위 모델의 Lift
####################################################################
def test_완벽한_모델은_검사율_1퍼센트에서_포착률_100퍼센트():
    """불량이 1,000개 중 10개(1%)이고 모델이 그 10개를 정확히 1~10위로
    올렸다면, 검사율 1%에서 모두 잡으므로 포착률 100%, Lift 100배다."""
    n, n_def = 1000, 10
    y = np.zeros(n, dtype=int); y[:n_def] = 1
    score = np.zeros(n); score[:n_def] = 1.0  # 불량에만 높은 점수

    t = threshold_table(y, score, alert_rates=(0.01,))
    row = t.iloc[0]
    assert row["포착_불량"] == 10
    assert row["포착률"] == pytest.approx(1.0)
    assert row["실제_검사율"] == pytest.approx(0.01)
    assert row["Lift"] == pytest.approx(100.0)
    assert row["정밀도"] == pytest.approx(1.0)


def test_무작위_점수는_Lift가_평균_1():
    """무작위 점수의 Lift 기댓값은 정의상 1이다.

    주의 — 처음에는 시드 하나로 `1.0 ± 0.35`를 요구했는데 실패했음(1.46).
    코드가 아니라 허용오차가 잘못이었음. 검사율 1%에서 불량 1,200개 중
    기댓값 12개를 잡으므로 표준편차가 약 3.45개, 즉 Lift 의 표준편차가
    약 0.29다. 시드 하나로는 1.5배도 흔히 나온다(20개 시드 실측: 평균
    1.002, 표준편차 0.293, 범위 0.50~1.46).

    그래서 여러 시드의 **평균**으로 본다. 표준오차가 0.29/√12 ≈ 0.085 이므로
    0.12 안에 들어와야 한다.
    """
    lifts_1pct, lifts_5pct = [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 200_000
        y = (rng.random(n) < 0.006).astype(int)
        score = rng.random(n)  # 정답과 무관
        t = threshold_table(y, score, alert_rates=(0.01, 0.05))
        lifts_1pct.append(float(t.loc[0, "Lift"]))
        lifts_5pct.append(float(t.loc[1, "Lift"]))

    assert np.mean(lifts_1pct) == pytest.approx(1.0, abs=0.12)
    assert np.mean(lifts_5pct) == pytest.approx(1.0, abs=0.06)


def test_전수검사_행이_기준선으로_들어감():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    score = np.linspace(0, 1, 10)
    t = threshold_table(y, score, alert_rates=(0.5,))
    last = t.iloc[-1]
    assert last["목표_검사율"] == 1.0
    assert last["포착률"] == 1.0
    assert last["Lift"] == 1.0
    assert last["정밀도"] == pytest.approx(0.2)  # 불량률


####################################################################
# ② Lift 를 실제 검사율로 나누는가
####################################################################
def test_동점이_많아_검사율이_어긋나면_Lift가_실제값으로_계산됨():
    """점수가 거의 전부 동점이면 목표 1%를 노려도 실제로는 훨씬 많이 걸린다.

    여기서는 100개 중 90개가 같은 점수 0.0이고 10개가 1.0이다.
    목표 검사율 5%의 임계값은 1.0이 아니라 0.0이 되어 100개 전부가
    알람이 된다. 그때 Lift 는 목표(0.05)로 나누면 20배로 부풀고,
    실제 검사율(1.0)로 나누면 1.0이 된다. 후자가 맞다.
    """
    y = np.zeros(100, dtype=int); y[:10] = 1
    score = np.zeros(100); score[:10] = 0.0  # 전부 동점

    t = threshold_table(y, score, alert_rates=(0.05,))
    row = t.iloc[0]
    assert row["실제_검사율"] == pytest.approx(1.0)
    assert row["포착률"] == pytest.approx(1.0)
    assert row["Lift"] == pytest.approx(1.0)  # 목표로 나눴다면 20.0 이 나온다


####################################################################
# summary
####################################################################
def test_summary_불균형에서_정확도가_무의미함을_보임():
    """Dummy 모델을 두는 목적이 이것이다 — 정확도는 높고 포착률은 0."""
    rng = np.random.default_rng(3)
    n = 100_000
    y = (rng.random(n) < 0.006).astype(int)
    score = np.zeros(n)  # 전부 같은 점수 = 아무 정보 없음

    s = summary(y, score, at_alert_rate=0.01)
    assert s["불량률"] == pytest.approx(0.006, abs=0.001)
    # 전부 동점이라 전부 알람 → 정확도는 불량률과 같아지고 Lift 는 1
    assert s["Lift@1%"] == pytest.approx(1.0)
    assert np.isnan(s["MCC@1%"])  # 한쪽만 예측 → 구분해서 nan


def test_summary_완벽한_점수의_PR_AUC는_1():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    s = summary(y, score, at_alert_rate=0.2)
    assert s["PR_AUC"] == pytest.approx(1.0)


####################################################################
# ③ 임계값 안정성 — 실제 데이터에서 일어난 상황
####################################################################
def test_분포가_밀리면_고정임계값의_검사율이_목표를_벗어남():
    """실제 데이터에서 학습 0.730% / 검증 0.308% / 시험 0.407% 로
    기간별 불량률이 최대 2.4배 달랐다. 그러면 점수 분포도 밀린다.

    여기서는 시험 구간 점수를 검증보다 +0.3 밀어 놓는다. 검증에서 정한
    고정 임계값을 쓰면 알람이 목표보다 훨씬 많아져야 하고, 이동 분위수를
    쓰면 목표에 붙어 있어야 한다.
    """
    rng = np.random.default_rng(7)
    n = 50_000
    y_v = (rng.random(n) < 0.006).astype(int)
    s_v = rng.random(n) * 0.5           # 검증: 0 ~ 0.5
    y_t = (rng.random(n) < 0.004).astype(int)
    s_t = rng.random(n) * 0.5 + 0.3     # 시험: 0.3 ~ 0.8 (분포가 밀렸음)

    out = threshold_stability(y_v, s_v, y_t, s_t, alert_rates=(0.01,))
    row = out.iloc[0]

    # 고정 임계값은 목표 1%를 크게 넘겨야 한다
    assert row["고정_실제검사율"] > 0.05
    assert row["고정_검사율_오차배수"] > 5
    # 이동 분위수는 목표에 붙어 있어야 한다
    assert row["이동분위수_실제검사율"] == pytest.approx(0.01, abs=0.002)


def test_분포가_안_밀리면_고정임계값도_목표에_붙음():
    rng = np.random.default_rng(8)
    n = 50_000
    y_v = (rng.random(n) < 0.006).astype(int)
    s_v = rng.random(n)
    y_t = (rng.random(n) < 0.006).astype(int)
    s_t = rng.random(n)  # 같은 분포

    out = threshold_stability(y_v, s_v, y_t, s_t, alert_rates=(0.01,))
    assert out.loc[0, "고정_검사율_오차배수"] == pytest.approx(1.0, abs=0.15)


####################################################################
# 비용 곡선
####################################################################
def test_cost_curve_유출비용이_커지면_검사를_더_함():
    """r 이 커지면(유출이 비싸지면) 최적 검사율이 커지거나 같아야 한다."""
    n, n_def = 100_000, 600
    y = np.zeros(n, dtype=int); y[:n_def] = 1
    # 불량을 앞쪽에 몰아 두고, 점수를 순위로 줘서 단조로운 포착률 곡선을 만든다
    score = np.linspace(1.0, 0.0, n)

    t = threshold_table(y, score)
    c = cost_curve(t, ratios=(1, 10, 100, 10_000))

    rates = c["최적_검사율"].tolist()
    assert rates == sorted(rates), f"유출비용이 커질수록 검사율이 줄면 안 됨: {rates}"
    assert c.loc[0, "최적_검사율"] < c.loc[len(c) - 1, "최적_검사율"]


####################################################################
# 작업 메모
#   - 2026-10-02 작성. 동점 처리(②)가 제일 중요함. 실제 모델 점수에도
#     동점이 생길 수 있고, 그때 Lift 를 목표 검사율로 나누면 헤드라인
#     숫자가 그대로 거짓이 됨
####################################################################
