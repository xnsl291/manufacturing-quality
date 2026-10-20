####################################################################
# test_explain.py — 공정 기여도 집계가 맞는지 검증
#
# 노리는 것
#   ① 측정값 수 교란 — 실제 데이터에서 두 공정이 측정값 968개 중 53%를
#      차지함. 합계만 쓰면 그 둘이 무조건 이긴다. 세 방식이 그 차이를
#      실제로 드러내는지 본다.
#   ② 방문 여부 열을 공정에 붙이는지 — 측정값 없이 통과 시각만 남는
#      공정(실제 L3_S42·L3_S46)은 방문 여부 열 하나뿐이라, 이걸 놓치면
#      그 공정이 집계와 ablation 에서 통째로 사라진다.
#   ③ SHAP 값 자체가 맞는지 — 가법성(기여도 합 + 기준값 = 모델 출력)
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explain import (  # noqa: E402
    GLOBAL,
    ablation_table,
    compute_shap,
    drop_stations,
    feature_to_station,
    station_contribution,
    stratified_sample,
)


####################################################################
# ② 입력값 → 공정 대응
####################################################################
@pytest.mark.parametrize(
    "name, expected",
    [
        ("L0_S0_F0", "L0_S0"),
        ("L3_S36_F3938", "L3_S36"),
        ("visited_L0_S0", "L0_S0"),
        ("visited_L3_S42", "L3_S42"),  # 측정값 열이 없는 날짜 전용 공정
        ("visited_line_3", GLOBAL),
        ("visited_station_count", GLOBAL),
        ("start_time", GLOBAL),
        ("missing_ratio", GLOBAL),
        ("path_id", GLOBAL),
    ],
)
def test_feature_to_station(name, expected):
    assert feature_to_station(name) == expected


def test_drop_stations가_측정값과_방문여부를_함께_뺌():
    names = ["L0_S0_F0", "L0_S0_F2", "visited_L0_S0",
             "L3_S36_F1", "visited_L3_S36", "visited_L3_S42", "start_time"]
    keep = drop_stations(names, ["L0_S0", "L3_S42"])
    assert keep == ["L3_S36_F1", "visited_L3_S36", "start_time"]


####################################################################
# 층화 표본
####################################################################
def test_stratified_sample_불량은_전부_남김():
    y = np.zeros(10_000, dtype=int)
    y[::500] = 1  # 불량 20개
    idx = stratified_sample(y, n_neg=300)
    assert y[idx].sum() == 20
    assert len(idx) == 320
    assert (np.diff(idx) > 0).all()  # 정렬됨


def test_stratified_sample_양품이_모자라면_전부_씀():
    y = np.array([1, 0, 0, 1, 0])
    idx = stratified_sample(y, n_neg=100)
    assert list(idx) == [0, 1, 2, 3, 4]


####################################################################
# ① 측정값 수 교란을 세 방식이 드러내는가
####################################################################
def test_합계는_측정값_많은_공정이_이기고_평균은_뒤집음():
    """공정 BIG 은 측정값 10개가 각각 약하게(0.1) 기여하고,
    공정 SMALL 은 측정값 1개가 강하게(0.5) 기여한다.
    합계로는 BIG(1.0) > SMALL(0.5), 평균으로는 SMALL(0.5) > BIG(0.1).
    이 뒤집힘이 실제 데이터의 L1_S24·L1_S25 문제와 같은 구조다."""
    names = [f"L0_S1_F{i}" for i in range(10)] + ["L3_S9_F0"]
    sv = np.tile(np.array([0.1] * 10 + [0.5]), (50, 1))
    sv[::2] *= -1  # 부호가 섞여도 절댓값으로 집계해야 함

    out = station_contribution(sv, names, top_k=1).set_index("station_key")

    assert out.loc["L0_S1", "합계"] == pytest.approx(1.0)
    assert out.loc["L3_S9", "합계"] == pytest.approx(0.5)
    assert out.loc["L0_S1", "합계_순위"] == 1
    assert out.loc["L3_S9", "평균_순위"] == 1
    assert out.loc["L3_S9", "상위포함수"] == 1   # 가장 강한 단일 측정값이 SMALL 소속
    assert out.loc["L0_S1", "상위포함수"] == 0


def test_세_방식_모두_상위인_공정이_맨_위로():
    """CORE 는 측정값 3개가 모두 강해서 세 방식 모두 1위여야 한다.
    WIDE 는 합계만 크고, NARROW 는 평균만 크다. 최악순위로 정렬하면
    CORE 가 맨 위, 나머지는 최악순위가 2 이상이어야 한다."""
    names = ([f"L0_S0_F{i}" for i in range(3)]        # CORE
             + [f"L0_S1_F{i}" for i in range(20)]     # WIDE
             + ["L0_S2_F0"])                          # NARROW
    row = np.array([0.9] * 3 + [0.06] * 20 + [0.8])
    sv = np.tile(row, (10, 1))

    out = station_contribution(sv, names, top_k=4)
    assert out.iloc[0]["station_key"] == "L0_S0"
    assert out.iloc[0]["최악순위"] == 1
    assert (out.iloc[1:]["최악순위"] >= 2).all()


def test_방문여부만_있는_공정도_집계에_나타남():
    names = ["L0_S0_F0", "visited_L3_S42", "start_time"]
    sv = np.tile(np.array([0.1, 0.4, 0.2]), (5, 1))
    out = station_contribution(sv, names).set_index("station_key")
    assert "L3_S42" in out.index
    assert out.loc["L3_S42", "합계"] == pytest.approx(0.4)
    assert GLOBAL in out.index


####################################################################
# ③ SHAP 가법성
####################################################################
def test_compute_shap_가법성():
    """TreeSHAP 의 기여도 합 + 기준값은 모델의 원시 출력(log-odds)과
    같아야 한다. 이게 깨지면 기여도 값이 열과 어긋나 붙은 것이다."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)),
                     columns=["L0_S0_F0", "L0_S0_F2", "L3_S36_F1", "start_time"])
    X.iloc[::7, 1] = np.nan  # 결측도 섞어 둔다
    y = (X["L0_S0_F0"] + 0.5 * X["L3_S36_F1"] + rng.normal(0, 0.3, 400) > 0.8).astype(int)
    b = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 7, "seed": 1},
                  lgb.Dataset(X, label=y), num_boost_round=20)

    sv = compute_shap(b, X)
    assert sv.shape == X.shape
    base = b.predict(X, pred_contrib=True)[:, -1]  # LightGBM 내장 기여도의 기준값
    raw = b.predict(X, raw_score=True)
    np.testing.assert_allclose(sv.sum(axis=1) + base, raw, atol=1e-6)


####################################################################
# ④ 겹친 신호 — 2026-10-16 실제 데이터에서 발견한 현상
####################################################################
def _redundant_data(n: int = 20_000, seed: int = 3):
    """신호를 가진 값 A 와 그 복제본 B 를 서로 다른 공정에 둔다.
    실제 데이터의 '공정파생(전역)'과 '절벽 구간 공정'의 관계를 흉내 낸 것."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    X = pd.DataFrame({
        "L0_S0_F0": a,                                # 신호 원천
        "L3_S5_F0": a + rng.normal(0, 0.01, n),       # 거의 같은 복제본, 다른 공정
        "L1_S9_F0": rng.normal(size=n),               # 무관한 값
    })
    y = (a + rng.normal(0, 0.3, n) > 1.6).astype(int)  # 불량률 약 5%
    s = np.array(["train"] * int(n * .6) + ["valid"] * int(n * .2) + ["test"] * (n - int(n * .8)))
    return X, y, s == "train", s == "valid", s == "test"


def test_ablation_하나만_빼면_안_떨어지고_둘_다_빼야_무너짐():
    """실제 데이터에서 SHAP 상위 공정만 빼면 111→109건, 공정파생만 빼면
    111→109건이었는데 **둘 다 빼면 111→10건**이었음. 단독 제거만 해 보면
    "그 공정은 중요하지 않다"는 틀린 결론을 내게 된다. 이 테스트는
    ablation 표가 그 차이를 드러내는지 확인한다."""
    X, y, tr, va, te = _redundant_data()
    out = ablation_table(X, y, tr, va, te, [
        ("전체", []),
        ("A만 제거", ["L0_S0"]),
        ("B만 제거", ["L3_S5"]),
        ("둘 다 제거", ["L0_S0", "L3_S5"]),
    ]).set_index("조건")

    # 판정은 포착 건수가 아니라 Lift 로 한다. 이 합성 데이터는 입력이 3개뿐이라
    # 점수에 동점이 많고, 동점 때문에 실제 검사율이 1%를 넘어 조건마다 다르다
    # (처음엔 포착 건수로 판정했다가 51건 vs 39건으로 실패했음 — 검사율이 달랐음).
    assert (out["실제검사율"] > 0).all()
    base = out.loc["전체", "Lift@1%"]
    assert base > 5
    # 하나만 빼면 복제본이 대신해서 거의 그대로
    assert out.loc["A만 제거", "Lift@1%"] > base * 0.8
    assert out.loc["B만 제거", "Lift@1%"] > base * 0.8
    # 둘 다 빼면 신호가 사라져 무작위(1) 근처로 떨어진다
    assert out.loc["둘 다 제거", "Lift@1%"] < 3


def test_ablation_첫_조건이_기준선이_아니면_멈춤():
    X, y, tr, va, te = _redundant_data(n=2_000)
    with pytest.raises(ValueError, match="기준선"):
        ablation_table(X, y, tr, va, te, [("A만 제거", ["L0_S0"])])


def test_compute_shap_모양이_다르면_멈춤():
    """LightGBM 이 기준값 열 없이 돌려주거나(버전·모델 형식이 바뀔 때)
    열 수가 달라지면 조용히 한 칸 밀린 기여도를 쓰지 말고 멈춰야 한다."""
    X = pd.DataFrame(np.zeros((5, 2)), columns=["L0_S0_F0", "start_time"])

    class FakeBooster:
        def predict(self, X, pred_contrib=False):
            return np.zeros(X.shape)  # 기준값 열이 빠진 모양 (5, 2)

    with pytest.raises(ValueError, match="SHAP 모양"):
        compute_shap(FakeBooster(), X)


####################################################################
# 작업 메모
#   - 2026-10-16 작성. 처음엔 compute_shap 에 "목록이면 [1]"
#     분기가 있었는데 실제로 실행되지 않는 죽은 코드라서 변형 검사가
#     못 잡았음. 분기를 지우고 모양 검사로 바꿨고 그걸 위 테스트로 확인함
####################################################################
