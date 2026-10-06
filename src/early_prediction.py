####################################################################
# early_prediction.py — 예측을 얼마나 앞당기면 성능이 얼마나 떨어지나
#
# 왜 이 분석이 필요한가
#   모든 공정을 통과한 뒤(= 최종 검사 직전)에 예측하면 성능은 가장
#   높지만, 운영에서 쓸 수 있는 폭이 좁다. "어차피 곧 전수 검사할
#   건데 그 직전에 예측하는 게 무슨 의미인가"라는 질문에 답이 없다.
#
#   예측을 앞당기면 재작업·선별 투입 같은 실제 개입이 가능해진다.
#   대신 쓸 수 있는 측정값이 줄어 성능이 떨어진다. 그 맞교환을 곡선으로
#   내는 것이 이 분석이다. 프로젝트 성격이 "분류 성능"에서
#   **"언제 판단할 것인가"** 로 바뀐다.
#
# 설계를 고친 부분 — 라인 번호는 공정 순서가 아니다
#   설계 6절에 시점 B 를 "L3 진입 전(L0~L2만 사용)"으로 적어 두었다.
#   라인 번호가 공정 순서라고 가정한 것이었다.
#   2026-10-02 Week 1 측정 결과 실제 순서는 **L0·L1 → L3 → L2** 였다
#   (제품 안에서의 상대 시각 중앙값: L0_S0 0.00, L3_S36 3.76,
#    L2_S26 15.93, L2_S28 16.37). 즉 L2 가 가장 마지막이다.
#   "L0~L2"로 자르면 마지막 공정을 포함하고 중간 공정을 빼는 꼴이 된다.
#   그래서 **측정한 공정 진행도(상대 시각)로 자른다.**
#
# 누설을 막기 위해 다시 계산하는 것
#   공정 진행도 t 에서 자르면 그 시점에 **아직 모르는 값**을 쓰면 안 된다.
#     쓸 수 있음  시작 시각, t 이전 공정의 방문 여부·측정값·측정 개수
#     쓸 수 없음  종료 시각, 총 공정시간, 전체 방문 공정 수, 전체 경로
#   그래서 공정 파생 입력값을 시점마다 **restrict 해서 다시 만든다.**
#   전체 기준으로 만든 값을 그대로 쓰면 미래 정보가 들어간다.
####################################################################

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import column_map, measurement_counts, station_time_matrix  # noqa: E402
from src.metrics import summary, threshold_table  # noqa: E402
from src.train import LGB_PARAMS  # noqa: E402


####################################################################
# stations_up_to — 공정 진행도 t 이전에 지나는 공정만 고름
#
# 역할  예측 시점을 정의한다. 기준은 Week 1에서 측정한 "제품 안에서의
#       상대 시각 중앙값"이다.
# 방식  방문 제품이 적어 순서를 믿기 어려운 공정은 **포함시킨다.**
#       제외하면 그 공정을 지난 제품의 정보를 통째로 잃기 때문이고,
#       표본이 적어 성능에 미치는 영향도 작다. 대신 몇 개가 그런
#       공정인지 함께 돌려준다.
####################################################################
def stations_up_to(order: pd.DataFrame, cutoff: float) -> tuple[list[str], dict]:
    keep = order[order["상대시각_중앙값"] <= cutoff]
    low = keep[keep["순서_신뢰"] != "보통"]["station_key"].tolist()
    return keep["station_key"].tolist(), {
        "포함 공정 수": int(len(keep)),
        "전체 공정 수": int(len(order)),
        "순서 신뢰 낮은 공정": low,
    }


####################################################################
# build_restricted — 시점 t 에서 알 수 있는 것만으로 입력을 다시 만듦
#
# 역할  누설을 막는 핵심. 전체 기준으로 만든 공정 파생 입력값을 그대로
#       쓰면 "아직 안 지난 공정을 지났는지"와 "총 공정시간"이 들어간다.
# 방식  허용된 공정의 컬럼만으로 방문 여부·측정 개수·경과 시간을
#       다시 계산한다. 종료 시각과 총 공정시간은 만들지 않는다.
#       대신 **그 시점까지의 경과 시간**을 쓴다.
####################################################################
def build_restricted(
    numeric_parquet: Path,
    date_parquet: Path,
    allowed: list[str],
) -> pd.DataFrame:
    nmap = column_map(numeric_parquet)
    dmap = column_map(date_parquet)
    nmap_a = nmap[nmap["station_key"].isin(allowed)]
    dmap_a = dmap[dmap["station_key"].isin(allowed)]

    visited, first = station_time_matrix(date_parquet, dmap_a)
    start = first.min(axis=1)
    elapsed = first.max(axis=1) - start  # 그 시점까지의 경과 시간 (총 공정시간이 아님)

    counts = measurement_counts(numeric_parquet, nmap_a)

    line_of = dmap_a.drop_duplicates("station_key").set_index("station_key")["line"]
    lines = sorted(line_of.unique())
    line_visited = pd.DataFrame(
        {
            f"visited_line_{ln}": visited[[s for s in visited.columns if line_of[s] == ln]]
            .any(axis=1)
            .astype("int8")
            for ln in lines
        }
    )

    raw = pq.read_table(numeric_parquet, columns=nmap_a["column"].tolist()).to_pandas()
    proc = pd.concat(
        [
            visited.astype("int8").add_prefix("visited_"),
            line_visited,
            pd.DataFrame(
                {
                    "visited_station_count": visited.sum(axis=1).astype("int16"),
                    "visited_line_count": line_visited.sum(axis=1).astype("int8"),
                    "start_time": start.astype("float32"),
                    "elapsed_so_far": elapsed.astype("float32"),
                }
            ),
            counts,
        ],
        axis=1,
    )
    return pd.concat([raw, proc], axis=1)


####################################################################
# main — 시점을 여러 개 잡아 성능 손실 곡선을 만듦
####################################################################
def main() -> int:
    ap = argparse.ArgumentParser(description="예측 시점별 성능 손실 곡선")
    ap.add_argument("--numeric", default="data/interim/train_numeric.parquet", type=Path)
    ap.add_argument("--date", default="data/interim/train_date.parquet", type=Path)
    ap.add_argument("--features", default="data/interim/process_features.parquet", type=Path)
    ap.add_argument("--order", default="reports/T_공정순서추정.csv", type=Path)
    ap.add_argument("--report-dir", default="reports", type=Path)
    args = ap.parse_args()

    order = pd.read_csv(args.order)
    feats = pd.read_parquet(args.features)
    label = pq.read_table(args.numeric, columns=["Id", "Response"]).to_pandas()
    assert (feats["Id"].to_numpy() == label["Id"].to_numpy()).all(), "Id 순서가 어긋남"
    y = label["Response"].to_numpy()
    split = feats["split"].to_numpy()
    tr, va, te = split == "train", split == "valid", split == "test"

    # 실제로 측정된 상대 시각 분포를 보고 자를 지점을 정한다
    med = order["상대시각_중앙값"]
    cutoffs = [
        ("① 첫 공정 직후", float(med.min())),
        ("② 전반부", 1.0),
        ("③ 중반", 4.0),
        ("④ 후반", 12.0),
        ("⑤ 전 공정 통과 (최종검사 직전)", float(med.max())),
    ]

    rows, meta = [], {}
    for name, cut in cutoffs:
        allowed, info = stations_up_to(order, cut)
        if not allowed:
            continue
        print(f"\n[{name}] 상대시각 ≤ {cut} / 공정 {info['포함 공정 수']}/{info['전체 공정 수']}개", flush=True)
        X = build_restricted(args.numeric, args.date, allowed)
        p = dict(LGB_PARAMS); p["metric"] = "average_precision"
        t0 = time.time()
        b = lgb.train(
            p, lgb.Dataset(X.loc[tr], label=y[tr]), num_boost_round=1000,
            valid_sets=[lgb.Dataset(X.loc[va], label=y[va])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        s_te = b.predict(X.loc[te])
        s = summary(y[te], s_te)
        rows.append(
            {
                "예측시점": name,
                "상대시각_컷": cut,
                "포함_공정수": info["포함 공정 수"],
                "입력수": int(X.shape[1]),
                "PR_AUC": s["PR_AUC"],
                "포착률@1%": s["포착률@1%"],
                "Lift@1%": s["Lift@1%"],
                "포착률@5%": float(
                    threshold_table(y[te], s_te, alert_rates=(0.05,)).loc[0, "포착률"]
                ),
                "라운드": int(b.best_iteration or 1000),
                "학습초": round(time.time() - t0, 1),
            }
        )
        meta[name] = info
        print(f"  PR-AUC {s['PR_AUC']:.4f} / 포착률@1% {s['포착률@1%']:.2%} "
              f"/ Lift {s['Lift@1%']:.1f} / 입력 {X.shape[1]}개", flush=True)
        del X

    out = pd.DataFrame(rows)
    full = out["포착률@1%"].iloc[-1]
    out["최종시점_대비_포착률"] = (out["포착률@1%"] / full).round(3)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.report_dir / "T_조기예측곡선.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 76)
    print(out.to_string(index=False))
    (Path("runs") / "early_prediction.json").write_text(
        json.dumps({"표": rows, "시점정보": meta}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{args.report_dir}/T_조기예측곡선.csv 저장함")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-06 작성. 설계 6절의 "L0~L2만 사용"을 공정 진행도 기준으로
#     바꿨음. 라인 번호가 공정 순서가 아니라는 Week 1 측정 때문임
#   - 종료 시각·총 공정시간은 시점마다 다시 계산해 넣지 않음. 그 시점에
#     아직 모르는 값이기 때문
####################################################################
