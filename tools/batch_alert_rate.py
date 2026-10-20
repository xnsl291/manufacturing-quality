####################################################################
# batch_alert_rate.py — "검사율 1%"를 하루·교대 단위로 다시 잼 (Week 6)
#
# 왜 이 실험을 하는가
#   README·모델카드의 대표 숫자(검사율 1%에서 111건, 11.5배)는 시험
#   구간 전체 236,750대를 한 묶음으로 보고 그 안에서 상위 1%를 자른
#   값이다. 정답을 본 것은 아니지만, 실제 현장은 약 5개월치를 다 모은
#   뒤 순위를 매기지 않는다. 하루나 교대가 끝날 때 그 묶음 안에서
#   고르거나, 제품이 나오는 즉시 미리 정한 문턱값과 비교한다.
#   그 단위로 바꾸면 숫자가 얼마나 달라지는지 잰다.
#
# 세 가지 운영 방식을 비교한다
#   전체 묶음     지금 대표 숫자. 시험 구간 전체 상위 1%
#   묶음 안 순위  하루(또는 교대)가 끝나면 그 묶음 안에서 상위 1%
#   전날 문턱값   직전 묶음의 상위 1% 점수를 문턱값으로 정해 두고,
#                 오늘 나오는 제품을 그 즉시 비교함. 미래 점수를 전혀
#                 보지 않는 유일한 방식이라 검사 물량이 흔들림
#
# 검사량이 다른 방식끼리 잡은 건수를 바로 비교하지 않는다
#   전날 문턱값은 검사량이 목표와 다르게 나온다. 많이 보면 많이 잡는
#   것은 당연하므로, 표에 "같은 대수를 전체 순위로 뽑았으면 몇 건"을
#   함께 둔다. 그 차이가 순위 품질의 차이고, 나머지는 검사량 차이다
#
# 시간 단위 — 데이터에 적혀 있지 않음
#   start_time 은 익명화된 숫자다. 0.1단위 제품 수의 자기상관이 2.4단위와
#   16.8단위에서 높아 각각 하루와 1주로 해석한다 **[추정]**. 그러면
#   1단위 ≈ 10시간, 8시간 교대 ≈ 0.8단위다. 묶음 경계는 검증 구간 첫
#   시각에서 시작하므로 실제 자정·교대 교체 시각과 맞는다는 보장은 없다.
#   그래서 결과 표에 단위 폭을 원래 숫자로 함께 남긴다.
#
# 불확실성 구간 — 하루 단위 블록 부트스트랩
#   제품 하나하나를 다시 뽑으면 같은 날 제품끼리 비슷하다는 점을 무시해
#   구간이 지나치게 좁아진다. 그래서 하루 묶음을 통째로 다시 뽑는다.
#   알람 여부는 원래 결과에 고정해 두고 다시 뽑으므로 "이 문턱값으로
#   다른 날들을 만났다면"의 구간이며, 모델을 다시 학습한 구간은 아니다.
#   하루를 넘는 상관(요일 패턴)은 반영하지 않으므로 1주 블록으로도 한 번
#   더 돌려 민감도를 본다.
#
# 라이선스 때문에 지키는 것
#   제품 단위 표는 만들지 않는다. 결과는 방식별 합계와 구간뿐이다
####################################################################

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import threshold_at_alert_rate  # noqa: E402

ALERT_RATE = 0.01
# 1주 = 16.8단위로 본 추정값. 하루 = 16.8 / 7, 교대 = 하루 / 3
WEEK = 16.8
WIDTHS = {"0.8단위(교대 추정)": WEEK / 21, "2.4단위(하루 추정)": WEEK / 7, "16.8단위(1주 추정)": WEEK}
DAY = "2.4단위(하루 추정)"
SMALL = 100  # 이보다 적은 묶음은 검사율 분포에서 뺌. 1대 묶음에서 1대를 보면 100%가 됨
N_BOOT = 2000
SEED = 0


####################################################################
# assign_batches — 시작 시각을 폭 width 의 묶음 번호로 바꿈
#
# 역할  같은 번호 = 같은 날(또는 교대). 기준점 t0 는 부르는 쪽이 정한다.
#       검증·시험을 이어 붙여 쓸 때 두 구간의 묶음 경계가 같아야 하기
#       때문이다
####################################################################
def assign_batches(start_time: np.ndarray, width: float, t0: float) -> np.ndarray:
    return np.floor((np.asarray(start_time, dtype="float64") - t0) / width).astype(int)


####################################################################
# alert_within_batch — 묶음마다 그 안에서 상위 a 를 알람
#
# 방식  묶음마다 threshold_at_alert_rate 를 따로 부른다. 전체 묶음 방식과
#       같은 함수라 차이는 묶는 단위에서만 생긴다
####################################################################
def alert_within_batch(score: np.ndarray, batch: np.ndarray, a: float) -> np.ndarray:
    alert = np.zeros(len(score), dtype=bool)
    for b in np.unique(batch):
        m = batch == b
        alert[m] = score[m] >= threshold_at_alert_rate(score[m], a)
    return alert


####################################################################
# alert_prev_batch — 직전 묶음의 상위 a 문턱값으로 오늘을 판정
#
# 역할  미래를 보지 않는 운영 방식. 제품이 나오자마자 판정할 수 있다
# 방식  묶음을 시간 순으로 돌며 바로 앞 묶음(제품이 있는 묶음)의 문턱값을
#       쓴다. 맨 앞 묶음은 prev_score 로 정한다. 묶음 사이에 생산이 없는
#       빈 기간이 있으면 그 앞 묶음을 쓴다
####################################################################
def alert_prev_batch(
    score: np.ndarray, batch: np.ndarray, a: float, prev_score: np.ndarray
) -> np.ndarray:
    alert = np.zeros(len(score), dtype=bool)
    thr = threshold_at_alert_rate(prev_score, a)
    for b in np.unique(batch):
        m = batch == b
        alert[m] = score[m] >= thr
        thr = threshold_at_alert_rate(score[m], a)
    return alert


####################################################################
# block_bootstrap — 묶음을 통째로 다시 뽑아 합계의 95% 구간을 냄
#
# 역할  y·alert 를 묶음별 (불량 수, 잡은 수, 검사 수, 제품 수)로 접은 뒤
#       묶음을 복원 추출한다. alerts 에 둘을 넘기면 같은 묶음 표본으로
#       차이를 재므로 짝지은 비교가 된다
# 반환  {이름: (점추정, 2.5%, 97.5%)} — 각 방식의 포착률·실제 검사율,
#       alerts 가 둘이면 첫째 − 둘째 잡은 건수 차이
####################################################################
def block_bootstrap(
    y: np.ndarray, block: np.ndarray, alerts: dict[str, np.ndarray], n_boot: int, seed: int
) -> dict[str, tuple[float, float, float]]:
    _, inv = np.unique(block, return_inverse=True)
    pos = np.bincount(inv, weights=y)
    cnt = np.bincount(inv)
    tp = {k: np.bincount(inv, weights=y * v) for k, v in alerts.items()}
    na = {k: np.bincount(inv, weights=v.astype(float)) for k, v in alerts.items()}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pos), size=(n_boot, len(pos)))
    out = {}
    for k in alerts:
        out[f"{k}_포착률"] = (tp[k].sum() / pos.sum(), tp[k][idx].sum(1) / pos[idx].sum(1))
        out[f"{k}_실제검사율"] = (na[k].sum() / cnt.sum(), na[k][idx].sum(1) / cnt[idx].sum(1))
    names = list(alerts)
    if len(names) == 2:
        a, b = names
        out[f"{a}−{b}_잡은건수차"] = (tp[a].sum() - tp[b].sum(), tp[a][idx].sum(1) - tp[b][idx].sum(1))
    return {k: (float(p), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
            for k, (p, v) in out.items()}


####################################################################
# summarize — 방식 하나의 합계와 묶음별 검사율 분포를 한 행으로
#
# 역할  "같은 검사량 전체 순위" 열을 여기서 같이 계산한다. 검사량이 다른
#       방식끼리 잡은 건수를 비교하지 않도록, 같은 대수를 시험 구간 전체
#       점수 순위로 뽑았을 때 잡는 건수를 옆에 둔다
####################################################################
def summarize(name: str, width: float | None, y: np.ndarray, score: np.ndarray,
              alert: np.ndarray, batch: np.ndarray | None) -> dict:
    n_alert = int(alert.sum())
    caught = int(y[alert].sum())
    rate = n_alert / len(y)
    recall = caught / int(y.sum())
    top = np.argsort(-score, kind="stable")[:n_alert]
    row = {"방식": name, "묶음폭_원단위": width, "묶음수": None, "작은묶음수": None,
           "묶음별_검사율_중앙값": None, "묶음별_검사율_상위10%": None, "묶음별_검사율_최대": None,
           "검사_대수": n_alert, "실제_검사율": rate, "포착_불량": caught,
           "같은검사량_전체순위_포착": int(y[top].sum()), "포착률": recall,
           "Lift": recall / rate if rate else float("nan"),
           "정밀도": caught / n_alert if n_alert else float("nan")}
    if batch is not None:
        g = pd.DataFrame({"a": alert, "b": batch}).groupby("b")["a"].agg(["mean", "size"])
        big = g.loc[g["size"] >= SMALL, "mean"]
        row.update({"묶음수": len(g), "작은묶음수": int((g["size"] < SMALL).sum()),
                    "묶음별_검사율_중앙값": float(big.median()),
                    "묶음별_검사율_상위10%": float(big.quantile(0.9)), "묶음별_검사율_최대": float(big.max())})
    return row


####################################################################
# time_unit_evidence — 시간 단위를 추정한 근거(자기상관)를 숫자로 남김
####################################################################
def time_unit_evidence(t: np.ndarray) -> dict:
    b = np.floor(t * 10).astype(int)
    c = np.bincount(b - b.min()).astype(float)
    c -= c.mean()
    ac = np.correlate(c, c, "full")[len(c) - 1:]
    ac /= ac[0]
    lag = int(np.argmax(ac[50:400]) + 50)
    return {"5~40단위_자기상관_최대_주기": lag / 10, "그때_자기상관": round(float(ac[lag]), 3),
            "2.4단위_자기상관": round(float(ac[24]), 3), "8.4단위_자기상관": round(float(ac[84]), 3)}


def main() -> int:
    run = ROOT / "runs" / "week2"
    feats = pd.read_parquet(ROOT / "data/interim/process_features.parquet", columns=["start_time", "split"])
    y_all = pq.read_table(ROOT / "data/interim/train_numeric.parquet", columns=["Response"]).column(0).to_numpy()
    va = feats["split"].to_numpy() == "valid"
    te = feats["split"].to_numpy() == "test"
    t_va, t_te = feats["start_time"].to_numpy()[va], feats["start_time"].to_numpy()[te]
    y = y_all[te].astype(int)
    s_va, s_te = np.load(run / "score_valid.npy"), np.load(run / "score_test.npy")
    s_lr = np.load(run / "score_test_logistic.npy")
    assert len(s_te) == len(y) == len(s_lr) and len(s_va) == va.sum()
    assert not np.isnan(t_te).any(), "시험 구간에 시작 시각 없는 제품이 있음"

    unit = time_unit_evidence(feats["start_time"].dropna().to_numpy())
    print(f"[시간 단위] {unit}")

    t0 = float(np.nanmin(t_va))
    a_all = s_te >= threshold_at_alert_rate(s_te, ALERT_RATE)
    rows = [summarize("전체 묶음(현재 대표 숫자)", None, y, s_te, a_all, None)]
    batches = {}
    for name, w in WIDTHS.items():
        bt, bv = assign_batches(t_te, w, t0), assign_batches(t_va, w, t0)
        # 검증·시험 경계 시각에 두 구간 제품이 같이 있어 마지막 검증 묶음과 첫
        # 시험 묶음 번호가 같을 수 있음. "전날"은 첫 시험 묶음보다 앞 번호로 잡음
        prev = s_va[bv == bv[bv < bt.min()].max()]
        a_in = alert_within_batch(s_te, bt, ALERT_RATE)
        rows.append(summarize(f"{name} 묶음 안 순위", w, y, s_te, a_in, bt))
        rows.append(summarize(f"{name} 전날 문턱값", w, y, s_te, alert_prev_batch(s_te, bt, ALERT_RATE, prev), bt))
        batches[name] = (bt, a_in)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "reports/T_묶음단위_검사율.csv", index=False, encoding="utf-8-sig")
    print("\n[운영 묶음 단위별 검사율 1% 결과 — LightGBM, 시험 구간]")
    print(table.to_string(index=False))

    day, a_day = batches[DAY]
    a_all_lr = s_lr >= threshold_at_alert_rate(s_lr, ALERT_RATE)
    a_day_lr = alert_within_batch(s_lr, day, ALERT_RATE)
    week = batches["16.8단위(1주 추정)"][0]
    pairs = [("하루", day, {"LightGBM_전체": a_all, "LightGBM_하루순위": a_day}),
             ("하루", day, {"LightGBM_전체": a_all, "Logistic_전체": a_all_lr}),
             ("하루", day, {"LightGBM_하루순위": a_day, "Logistic_하루순위": a_day_lr}),
             ("1주(민감도)", week, {"LightGBM_전체": a_all, "Logistic_전체": a_all_lr})]
    ci_rows = []
    for unit_name, blk, alerts in pairs:
        for k, (p, lo, hi) in block_bootstrap(y, blk, alerts, N_BOOT, SEED).items():
            ci_rows.append({"다시뽑는단위": unit_name, "항목": k, "점추정": p, "하한_2.5%": lo, "상한_97.5%": hi})
    ci = pd.DataFrame(ci_rows).drop_duplicates(["다시뽑는단위", "항목"])
    ci.to_csv(ROOT / "reports/T_부트스트랩_구간.csv", index=False, encoding="utf-8-sig")
    print(f"\n[블록 부트스트랩 {N_BOOT}회, 95% 구간] 하루 묶음 {len(np.unique(day))}개, 1주 묶음 {len(np.unique(week))}개")
    print(ci.to_string(index=False))

    (ROOT / "runs/batch_alert_rate.json").write_text(json.dumps(
        {"검사율": ALERT_RATE, "시간단위_근거": unit, "묶음폭": WIDTHS, "작은묶음_기준_대수": SMALL,
         "부트스트랩": {"횟수": N_BOOT, "시드": SEED, "하루묶음수": int(len(np.unique(day))),
                     "1주묶음수": int(len(np.unique(week)))},
         "표": table.to_dict("records"), "구간": ci.to_dict("records")},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n표를 reports/T_묶음단위_검사율.csv, reports/T_부트스트랩_구간.csv 에 저장함 (합계만)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-20 작성. 사용자가 W6 보고의 "하루·교대 단위로 다시 잴까요,
#     불확실성 구간도 같이 할까요"에 "진행"으로 답해 만듦
#   - 같은 날 맥락 없는 검토 에이전트 지적으로 같은검사량 열, 작은 묶음
#     제외 분포, 검증·시험 경계 처리, 짝지은 전체−하루 구간, 1주 블록
#     민감도를 더함
#   - 1주 = 16.8단위는 자기상관으로 고른 추정임. 실제 단위는 대회 자료에
#     없음
####################################################################
