####################################################################
# test_batch_alert_rate.py — 묶음 단위 검사율 실험이 맞게 도는지 검증
#
# 노리는 것
#   ① 묶음 안 순위가 정말 묶음마다 따로 상위 a 를 고르는지. 전체에서
#      한 번 고르면 "하루 단위로 재도 거의 같다"는 결론이 저절로 나옴
#   ② 전날 문턱값이 오늘 점수를 보지 않는지. 오늘을 보면 미래 정보를
#      쓰는 것이라 검사 물량 흔들림이 숨겨짐
#   ③ 부트스트랩이 날짜를 통째로 뽑는지, 시드가 같으면 같은 구간인지
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.batch_alert_rate import (  # noqa: E402
    alert_prev_batch,
    alert_within_batch,
    assign_batches,
    block_bootstrap,
)


def test_assign_batches_경계가_t0_기준():
    t = np.array([10.0, 10.9, 11.0, 12.5])
    assert assign_batches(t, 1.0, 10.0).tolist() == [0, 0, 1, 2]


def test_alert_within_batch_묶음마다_따로_고름():
    # 묶음 0 은 점수가 전부 높고 묶음 1 은 전부 낮음. 전체에서 고르면
    # 묶음 0 에서만 알람이 나오지만, 묶음마다 고르면 양쪽에서 하나씩 나와야 함
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.05, 0.04, 0.03])
    batch = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    alert = alert_within_batch(score, batch, 0.25)
    assert alert.tolist() == [True, False, False, False, True, False, False, False]


def test_alert_prev_batch_오늘_점수를_보지_않음():
    # 묶음 1 의 판정은 묶음 0 의 문턱값(상위 50% = 0.5 이상)으로만 정해짐.
    # 묶음 1 점수가 전부 문턱값보다 낮으므로 알람이 하나도 없어야 함
    score = np.array([0.6, 0.4, 0.3, 0.2])
    batch = np.array([0, 0, 1, 1])
    prev = np.array([0.5, 0.5])
    alert = alert_prev_batch(score, batch, 0.5, prev)
    assert alert.tolist() == [True, False, False, False]


def test_alert_prev_batch_첫_묶음은_prev_score_로_정함():
    score = np.array([0.6, 0.4])
    batch = np.array([0, 0])
    alert = alert_prev_batch(score, batch, 0.5, prev_score=np.array([0.9, 0.9]))
    assert not alert.any()


def test_alert_prev_batch_빈_기간은_그_앞_묶음을_씀():
    # 묶음 1 이 비어 있음. 묶음 2 는 묶음 0 의 문턱값(0.5)으로 판정해야 함
    score = np.array([0.6, 0.4, 0.55, 0.45])
    batch = np.array([0, 0, 2, 2])
    alert = alert_prev_batch(score, batch, 0.5, prev_score=np.array([0.0, 0.0]))
    assert alert.tolist() == [True, True, True, False]


def test_block_bootstrap_같은_시드면_같고_차이는_짝지음():
    rng = np.random.default_rng(1)
    n = 2000
    day = np.repeat(np.arange(20), n // 20)
    y = (rng.random(n) < 0.05).astype(int)
    a = rng.random(n) < 0.1
    ci1 = block_bootstrap(y, day, {"A": a, "B": a}, 200, 0)
    ci2 = block_bootstrap(y, day, {"A": a, "B": a}, 200, 0)
    assert ci1 == ci2
    # 같은 알람을 두 번 넘기면 짝지은 차이는 모든 표본에서 0
    assert ci1["A−B_잡은건수차"] == (0.0, 0.0, 0.0)
    p, lo, hi = ci1["A_포착률"]
    assert p == a[y == 1].mean() and lo <= p <= hi


def test_block_bootstrap_묶음을_통째로_뽑음():
    # 묶음이 둘뿐이면 다시 뽑은 포착률은 {0번만, 1번만, 섞음} 세 값만 나와야 함.
    # 제품 단위로 뽑으면 그 사이 값들이 나옴
    y = np.array([1, 1, 0, 1, 1, 0])
    a = np.array([True, True, False, False, False, False])
    block = np.array([0, 0, 0, 1, 1, 1])
    _, lo, hi = block_bootstrap(y, block, {"A": a}, 500, 0)["A_포착률"]
    assert {lo, hi} <= {0.0, 0.5, 1.0}
    assert lo == 0.0 and hi == 1.0
