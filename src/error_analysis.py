####################################################################
# error_analysis.py — 모델이 어떤 상황에서 틀리는지 봄 (Week 3)
#
# 왜 필요한가
#   "성능이 몇 %인가"보다 **"어떤 상황에서 틀리는지 알고 있다"**가
#   현업 신뢰를 만든다. 품질 담당자가 알람 목록을 받아 쓰려면
#   "이 모델은 이런 제품을 놓친다"를 미리 알아야 한다.
#
# 네 집단으로 나눠 비교한다
#   TP  알람했고 실제 불량      — 잘 잡은 것
#   FN  알람 안 했는데 실제 불량 — **놓친 것. 가장 중요**
#   FP  알람했는데 실제 양품    — 헛수고한 것
#   TN  알람 안 했고 실제 양품  — 정상
#
# 라이선스 때문에 지키는 것
#   대회 규정이 데이터 재배포를 금지한다. 제품 하나하나의 측정값이
#   들어간 표는 **원본 행과 같아서 저장소에 남기지 않는다.** 이 파일은
#   집단별 **집계값만** 낸다. 개별 제품 확인은 화면(Streamlit)에서
#   로컬 파일을 읽어 하고, 공개하지 않는다.
####################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import threshold_at_alert_rate  # noqa: E402

# 집단별로 비교할 공정 파생 입력값. 측정값 자체는 익명이라 비교해도
# 해석할 수 없으므로, 뜻을 말할 수 있는 공정 구조 값만 쓴다.
PROFILE_COLS = [
    "visited_station_count",
    "visited_line_count",
    "measurement_count",
    "missing_ratio",
    "duration",
    "start_time",
]


####################################################################
# classify — 임계값 하나로 네 집단으로 나눔
####################################################################
def classify(y_true: np.ndarray, score: np.ndarray, alert_rate: float) -> pd.Series:
    thr = threshold_at_alert_rate(score, alert_rate)
    alert = score >= thr
    y = np.asarray(y_true).astype(bool)
    out = np.where(alert & y, "TP_잡음", np.where(~alert & y, "FN_놓침",
                   np.where(alert & ~y, "FP_헛수고", "TN_정상")))
    return pd.Series(out, name="집단")


####################################################################
# group_profile — 집단별 집계 비교표
#
# 역할  놓친 불량(FN)이 잡은 불량(TP)과 **체계적으로 다른지** 본다.
#       다르면 그 차이가 모델의 한계를 설명한다.
# 방식  중앙값으로 비교한다. 평균은 공정시간처럼 꼬리가 긴 값에서
#       한두 건에 끌려간다. 집단 크기를 같이 내서 표본이 적은 집단의
#       중앙값을 그대로 믿지 않게 한다.
####################################################################
def group_profile(groups: pd.Series, feats: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    cols = [c for c in PROFILE_COLS if c in feats.columns]
    df = feats[cols].copy()
    df["모델점수"] = score
    df["집단"] = groups.to_numpy()

    prof = df.groupby("집단").median(numeric_only=True)
    prof.insert(0, "제품수", df.groupby("집단").size())
    order = ["TP_잡음", "FN_놓침", "FP_헛수고", "TN_정상"]
    return prof.reindex([o for o in order if o in prof.index])


####################################################################
# path_mix — 집단별로 어떤 경로의 제품이 모였나
#
# 역할  놓친 불량이 특정 경로에 몰려 있는지 본다. 몰려 있으면
#       "이 경로의 제품은 모델이 잘 못 잡는다"고 말할 수 있고,
#       그건 현장에서 바로 쓸 수 있는 정보다.
# 방식  경로 번호별 비중을 집단별로 낸다. 경로 번호는 학습 구간 상위
#       20개와 기타(0)로 묶인 값이다.
####################################################################
def path_mix(groups: pd.Series, path_id: pd.Series, top: int = 6) -> pd.DataFrame:
    df = pd.DataFrame({"집단": groups.to_numpy(), "경로": path_id.to_numpy()})
    share = (
        df.groupby(["집단", "경로"]).size().unstack(fill_value=0).pipe(lambda d: d.div(d.sum(axis=1), axis=0))
    )
    # 전체에서 많이 나온 경로만 남긴다
    keep = share.sum(axis=0).nlargest(top).index
    out = share[keep]
    order = ["TP_잡음", "FN_놓침", "FP_헛수고", "TN_정상"]
    return out.reindex([o for o in order if o in out.index])


####################################################################
# period_mix — 집단별로 어느 생산 시기에 몰려 있나
#
# 역할  Week 1에서 구간별 불량률이 최대 2.7배 달랐다. 놓친 불량이
#       특정 시기에 몰려 있으면, 그건 모델이 아니라 **공정 상태 변화**가
#       원인이라는 뜻이고 재학습 주기 이야기로 이어진다.
####################################################################
def period_mix(groups: pd.Series, start_time: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    s = start_time.to_numpy()
    rank = np.argsort(np.argsort(np.nan_to_num(s, nan=-np.inf), kind="stable"))
    bins = (rank * n_bins // len(s)) + 1
    df = pd.DataFrame({"집단": groups.to_numpy(), "시기": bins})
    share = df.groupby(["집단", "시기"]).size().unstack(fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)
    order = ["TP_잡음", "FN_놓침", "FP_헛수고", "TN_정상"]
    return share.reindex([o for o in order if o in share.index])


def main() -> int:
    ap = argparse.ArgumentParser(description="오탐·미탐 집계 분석")
    ap.add_argument("--numeric", default="data/interim/train_numeric.parquet", type=Path)
    ap.add_argument("--features", default="data/interim/process_features.parquet", type=Path)
    ap.add_argument("--score", default="runs/week2/score_test.npy", type=Path)
    ap.add_argument("--alert-rate", default=0.01, type=float)
    ap.add_argument("--report-dir", default="reports", type=Path)
    args = ap.parse_args()

    if not args.score.exists():
        print(f"[중단] {args.score} 가 없음. 먼저 src/train.py 를 돌릴 것")
        return 1

    feats = pd.read_parquet(args.features)
    label = pq.read_table(args.numeric, columns=["Id", "Response"]).to_pandas()
    assert (feats["Id"].to_numpy() == label["Id"].to_numpy()).all(), "Id 순서가 어긋남"

    te = feats["split"].to_numpy() == "test"
    y = label["Response"].to_numpy()[te]
    score = np.load(args.score)
    assert len(score) == te.sum(), f"점수 {len(score)}개 != 시험 구간 {te.sum()}개"

    f_te = feats.loc[te].reset_index(drop=True)
    groups = classify(y, score, args.alert_rate)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    print(f"검사율 {args.alert_rate:.0%} 기준\n")

    prof = group_profile(groups, f_te, score)
    prof.to_csv(args.report_dir / "T_오탐미탐_집계.csv", encoding="utf-8-sig")
    print("[집단별 집계 — 중앙값]")
    print(prof.to_string())

    paths = path_mix(groups, f_te["path_id"])
    paths.to_csv(args.report_dir / "T_오탐미탐_경로비중.csv", encoding="utf-8-sig")
    print("\n[집단별 경로 비중 — 많이 나온 경로만]")
    print((paths * 100).round(1).to_string())

    periods = period_mix(groups, f_te["start_time"])
    periods.to_csv(args.report_dir / "T_오탐미탐_시기비중.csv", encoding="utf-8-sig")
    print("\n[집단별 생산 시기 비중 — 시험 구간을 5등분]")
    print((periods * 100).round(1).to_string())

    # 놓친 불량이 잡은 불량과 얼마나 다른지 한 줄로
    if {"TP_잡음", "FN_놓침"} <= set(prof.index):
        diff = {}
        for c in prof.columns:
            if c in ("제품수",):
                continue
            tp, fn = prof.loc["TP_잡음", c], prof.loc["FN_놓침", c]
            if pd.notna(tp) and pd.notna(fn) and tp != 0:
                diff[c] = round(float(fn / tp), 3)
        print("\n[놓친 불량 ÷ 잡은 불량 — 1에서 멀수록 체계적 차이]")
        for k, v in sorted(diff.items(), key=lambda kv: abs(np.log(max(kv[1], 1e-9))), reverse=True):
            print(f"  {k}: {v}")
        (Path("runs") / "error_analysis.json").write_text(
            json.dumps({"검사율": args.alert_rate, "집단크기": prof["제품수"].to_dict(),
                        "FN_대_TP_비": diff}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\n표를 {args.report_dir}/ 아래에 저장함 (집계만. 개별 제품 표는 만들지 않음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-06 작성. 개별 제품 표를 만들지 않는 것은 대회 규정의
#     데이터 재배포 금지 때문임. 개별 확인은 화면에서 로컬로만 함
####################################################################
