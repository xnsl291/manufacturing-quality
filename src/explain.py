####################################################################
# explain.py — 모델이 어느 공정의 값에 반응하는지 봄 (Week 4)
#
# 이번에 답할 질문
#   조기예측 곡선이 절벽이었음. 공정 26개까지는 무작위와 같고(Lift 1.1)
#   47개에서 전체 성능에 도달했음(11.3). 그렇다면 **그 사이에 들어오는
#   공정 중 어디가 신호를 가져오는가.** 이 파일은 그 구간의 공정을
#   SHAP 기여도 순위와 대조해 답한다.
#
# 공정 단위로 묶는 방식을 셋 다 내는 이유
#   측정값 수가 극단적으로 치우쳐 있음(L1_S25 284개, L1_S24 229개로 두
#   공정이 968개 중 53%. 중앙값은 8개). 그래서 어느 한 방식만 쓰면 틀림.
#     합계        측정값이 많은 공정이 무조건 이김
#     평균        측정값 1~2개짜리 공정이 과대평가될 수 있음
#     상위 포함 수 상위 N 이라는 컷이 임의적임
#   셋을 나란히 두고 **세 방식에서 일관되게 올라오는 공정**을 신호원으로 본다.
#
# 표현 제한 — 모든 산출물에 넣음
#   SHAP 은 "모델이 이 값에 크게 반응했다"는 뜻이지 "이 공정이 불량을
#   일으켰다"는 뜻이 아님. 측정값은 익명이라 무엇을 재는지도 모름.
#   그래서 공정 단위까지만 말하고, 현장 확인의 **우선순위**로만 제시한다.
#
# 표본 편향
#   시험 구간 불량은 964건(0.407%)뿐이라 무작위로 3만 행을 뽑으면 불량이
#   120건쯤만 들어온다. 그래서 **불량 전부 + 양품 표본**으로 층화한다.
#   그 결과 전역 기여도는 불량 쪽으로 기운 표본에서 계산된 값이 된다.
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
from src.ingest import parse_column  # noqa: E402
from src.metrics import summary  # noqa: E402
from src.train import LGB_PARAMS  # noqa: E402

GLOBAL = "공정파생(전역)"  # 특정 공정에 속하지 않는 입력값을 묶는 이름

CAUTION = (
    "이 값은 모델이 해당 공정의 값에 크게 반응했다는 뜻이며, 해당 공정이 불량을 "
    "일으켰다는 뜻이 아님. 측정값은 익명이라 무엇을 재는지 알 수 없음. "
    "현장 확인의 우선순위로만 사용할 것."
)


####################################################################
# feature_to_station — 입력값 이름을 공정 단위로 묶음
#
# 역할  공정 단위 집계와 ablation 이 모두 이 대응표를 쓴다.
# 방식  원시 측정값(L0_S0_F0)과 방문 여부(visited_L0_S0)는 그 공정으로
#       보낸다. 방문 여부를 빼먹으면, 측정값 없이 통과 시각만 남는
#       L3_S42·L3_S46 이 집계에서 통째로 사라진다.
#       라인 방문·공정 수·시각·측정 개수·경로 번호는 특정 공정에 속하지
#       않으므로 한 묶음(공정파생(전역))으로 보낸다.
####################################################################
def feature_to_station(name: str) -> str:
    got = parse_column(name)
    if got is not None:
        return got["station_key"]
    if name.startswith("visited_L"):
        rest = name[len("visited_"):]
        if parse_column(rest + "_F0") is not None:
            return rest
    return GLOBAL


####################################################################
# stratified_sample — 불량 전부 + 양품 표본의 행 번호를 고름
#
# 역할  SHAP 계산 비용을 줄이면서 불량 쪽 기여도를 충분히 본다.
# 방식  불량은 하나도 버리지 않는다. 양품만 n_neg 개 뽑는다.
#       양품이 n_neg 보다 적으면 전부 쓴다.
# 돌려주는 것  정렬된 행 번호 배열
####################################################################
def stratified_sample(y: np.ndarray, n_neg: int, seed: int = 42) -> np.ndarray:
    y = np.asarray(y)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    if len(neg) > n_neg:
        neg = rng.choice(neg, size=n_neg, replace=False)
    return np.sort(np.concatenate([pos, neg]))


####################################################################
# station_contribution — 측정값 단위 SHAP 을 공정 단위 표로 묶음
#
# 역할  세 방식을 한 표에 나란히 둔다. 순위도 각각 매겨서, 세 방식에서
#       모두 상위권인 공정이 바로 보이게 한다.
# 방식  측정값별 기여도 = 표본 전체에서 |SHAP| 의 평균.
#       합계 = 그 공정 측정값 기여도의 합
#       평균 = 합계 ÷ 그 공정 측정값 수
#       상위포함수 = 전체 측정값 중 기여도 상위 top_k 개 가운데 그 공정 소속 개수
####################################################################
def station_contribution(
    shap_values: np.ndarray, feature_names: list[str], top_k: int = 30
) -> pd.DataFrame:
    per_feature = pd.DataFrame(
        {
            "feature": feature_names,
            "station_key": [feature_to_station(f) for f in feature_names],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    top_feats = set(per_feature.nlargest(top_k, "mean_abs_shap")["feature"])
    per_feature["in_top"] = per_feature["feature"].isin(top_feats)

    g = per_feature.groupby("station_key")
    out = pd.DataFrame(
        {
            "입력수": g.size(),
            "합계": g["mean_abs_shap"].sum(),
            "상위포함수": g["in_top"].sum().astype(int),
        }
    )
    out["평균"] = out["합계"] / out["입력수"]
    for col in ("합계", "평균", "상위포함수"):
        out[f"{col}_순위"] = out[col].rank(ascending=False, method="min").astype(int)
    # 세 방식 중 가장 나쁜 순위 — 작을수록 "세 방식 모두에서 상위"
    out["최악순위"] = out[["합계_순위", "평균_순위", "상위포함수_순위"]].max(axis=1)
    return out.sort_values(["최악순위", "합계_순위"]).reset_index().rename(
        columns={"index": "station_key"}
    )


####################################################################
# drop_stations — 지정한 공정에 속한 입력값을 모두 뺀 열 목록
#
# 역할  ablation 에 쓴다. 원시 측정값뿐 아니라 방문 여부까지 함께 빼야
#       "그 공정 정보를 없앴다"고 말할 수 있다. 방문 여부를 남기면 모델이
#       그쪽으로 우회해 성능이 덜 떨어지고, SHAP 해석이 과소평가된다.
####################################################################
def drop_stations(feature_names: list[str], stations: list[str]) -> list[str]:
    drop = set(stations)
    return [f for f in feature_names if feature_to_station(f) not in drop]


####################################################################
# compute_shap — 입력값별 SHAP 기여도
#
# 2026-10-20부터 shap 라이브러리 대신 LightGBM 내장 계산(pred_contrib)을 씀.
# 계획(ADR-006)은 처음부터 내장 계산이었는데 구현 때 shap.TreeExplainer 를
# 썼음. 두 방식은 같은 알고리즘(TreeSHAP)이고, 실제 기여도 표본 30,964행
# 전체에서 값을 비교해 같은 것을 확인한 뒤 바꿨음. 의존성이 5개 줄어듦.
#
# 출력은 (행, 입력 수 + 1)이고 마지막 열이 기준값(base)임. 그 열을 떼고
# 돌려줌. 모양이 예상과 다르면(LightGBM 버전·모델 형식 변경) 멈춤 —
# 조용히 열이 하나 밀린 기여도를 쓰는 것보다 멈추는 것이 낫다.
####################################################################
def compute_shap(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    values = np.asarray(booster.predict(X, pred_contrib=True))
    if values.shape != (X.shape[0], X.shape[1] + 1):
        raise ValueError(
            f"SHAP 모양 {values.shape} 이 입력 {X.shape} + 기준값 열과 다름. "
            "LightGBM 버전이나 모델 형식이 바뀌었는지 확인할 것"
        )
    return values[:, :-1]


def fit_eval(X: pd.DataFrame, y: np.ndarray, tr, va, te) -> tuple[dict, int]:
    p = dict(LGB_PARAMS)
    p["metric"] = "average_precision"
    b = lgb.train(
        p,
        lgb.Dataset(X.loc[tr], label=y[tr]),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X.loc[va], label=y[va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return summary(y[te], b.predict(X.loc[te])), int(b.best_iteration or 1000)


####################################################################
# ablation_table — 공정 묶음을 빼고 재학습해 실제로 성능이 떨어지는지 봄
#
# 왜 묶음을 **함께** 빼 보는가 — 2026-10-16 실제 데이터에서 알게 된 것
#   SHAP 상위 3개 공정을 빼도 포착 건수가 111 → 109 로 그대로였음.
#   공정파생(전역)만 빼도 111 → 109. 그런데 **둘 다 빼면 111 → 10**,
#   무작위 수준(Lift 1.0)으로 무너졌음. 신호가 두 곳에 겹쳐 있어서
#   하나만 빼면 다른 쪽이 대신하는 것이었음.
#
#   즉 SHAP 순위도, 한 묶음씩 빼는 ablation 도 이런 중복을 드러내지
#   못한다. 그래서 단독 제거와 함께 제거를 같은 표에 둔다.
#
# 방식  conditions 는 (이름, 뺄 공정 목록) 쌍의 목록이다. 첫 행은 기준선
#       (아무것도 안 뺌)이어야 하고, 각 행에 기준선 대비 포착 건수 차이를 붙인다.
#       포착 건수는 실제로 잡은 불량 수라서 노이즈 크기(√건수)와 바로 비교된다.
#
# 주의 — 포착 건수는 실제 검사율이 같을 때만 비교됨 (2026-10-16 테스트에서 발견)
#   점수에 동점이 많으면 "상위 1%" 임계값에 걸리는 제품이 1%를 넘는다.
#   합성 테스트에서 시험 4,000행의 1%는 40대인데 51건을 잡았음. 그러면
#   조건마다 실제 검사율이 달라 포착 건수끼리 비교할 수 없다. 그래서
#   **실제 검사율을 같이 내고**, 판정은 실제 검사율로 나눈 Lift 로 한다.
####################################################################
def ablation_table(
    X: pd.DataFrame, y: np.ndarray, tr, va, te, conditions: list[tuple[str, list[str]]]
) -> pd.DataFrame:
    if not conditions or conditions[0][1]:
        raise ValueError("첫 조건은 아무것도 빼지 않는 기준선이어야 함")
    names = list(X.columns)
    n_def = int(y[te].sum())
    rows = []
    for name, drop in conditions:
        keep = drop_stations(names, drop)
        m, r = fit_eval(X[keep], y, tr, va, te)
        rows.append({"조건": name, "입력수": len(keep), "라운드": r,
                     "PR_AUC": m["PR_AUC"], "포착률@1%": m["포착률@1%"],
                     "실제검사율": m["실제검사율@1%"], "Lift@1%": m["Lift@1%"],
                     "포착건수": int(round(m["포착률@1%"] * n_def))})
        print(f"  {name}: 입력 {len(keep)} / 포착 {rows[-1]['포착건수']}건 "
              f"/ Lift {m['Lift@1%']:.1f}", flush=True)
    out = pd.DataFrame(rows)
    out["기준선대비_건수"] = out["포착건수"] - out.loc[0, "포착건수"]
    return out


####################################################################
# load_model_inputs — 저장된 모델이 학습한 그대로 입력 행렬을 다시 만듦
#
# 역할  SHAP 과 ablation 이 모두 "모델이 본 것과 똑같은 열 순서"를 전제한다.
#       순서가 하나라도 어긋나면 SHAP 값이 엉뚱한 열에 붙고, 오류 없이
#       틀린 공정 순위가 나온다. 그래서 열 구성을 확인하고 모델 순서로 맞춘다.
# 따로 둔 이유  같은 재구성을 탐색 중에 임시 스크립트 두 개에서 반복했음.
#       한 곳에 두고 Id 정렬 확인까지 함께 묶는다.
####################################################################
def load_model_inputs(
    model_path: Path, numeric: Path, features: Path
) -> tuple[lgb.Booster, pd.DataFrame, np.ndarray, np.ndarray]:
    booster = lgb.Booster(model_file=str(model_path))
    names = booster.feature_name()

    feats = pd.read_parquet(features)
    label = pq.read_table(numeric, columns=["Id", "Response"]).to_pandas()
    assert (feats["Id"].to_numpy() == label["Id"].to_numpy()).all(), "Id 순서가 어긋남"

    raw = pq.read_table(numeric, columns=[n for n in names if parse_column(n)]).to_pandas()
    X = pd.concat([raw, feats.drop(columns=["Id", "split"])], axis=1)
    if set(X.columns) != set(names):
        raise ValueError("입력 열 구성이 모델과 다름 — 공정 파생 입력을 다시 만들었는지 확인할 것")
    return booster, X[names], label["Response"].to_numpy(), feats["split"].to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="SHAP 공정 기여도 + ablation")
    ap.add_argument("--numeric", default="data/interim/train_numeric.parquet", type=Path)
    ap.add_argument("--features", default="data/interim/process_features.parquet", type=Path)
    ap.add_argument("--model", default="runs/week2/model_lgb.txt", type=Path)
    ap.add_argument("--order", default="reports/T_공정순서추정.csv", type=Path)
    ap.add_argument("--n-neg", default=30_000, type=int)
    ap.add_argument("--ablate-k", default=3, type=int)
    ap.add_argument("--report-dir", default="reports", type=Path)
    args = ap.parse_args()

    booster, X, y, split = load_model_inputs(args.model, args.numeric, args.features)
    names = list(X.columns)
    tr, va, te = split == "train", split == "valid", split == "test"

    # ---- SHAP -----------------------------------------------------------
    te_idx = np.flatnonzero(te)
    pick = te_idx[stratified_sample(y[te_idx], args.n_neg)]
    print(f"SHAP 표본 {len(pick):,}행 (불량 {int(y[pick].sum()):,} + 양품 "
          f"{len(pick) - int(y[pick].sum()):,})", flush=True)
    t0 = time.time()
    sv = compute_shap(booster, X.iloc[pick])
    print(f"  SHAP {time.time() - t0:.0f}초", flush=True)

    contrib = station_contribution(sv, names)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    contrib.to_csv(args.report_dir / "T_공정기여도.csv", index=False, encoding="utf-8-sig")

    # ---- 조기예측 절벽 구간과 대조 --------------------------------------
    order = pd.read_csv(args.order)
    band = order[(order["상대시각_중앙값"] > 1.0) & (order["상대시각_중앙값"] <= 4.0)]
    band_keys = set(band["station_key"])
    contrib["절벽구간"] = contrib["station_key"].isin(band_keys)

    # ---- ablation — 단독 제거와 함께 제거를 같은 표에 -------------------
    top = (
        contrib[contrib["station_key"] != GLOBAL]
        .nsmallest(args.ablate_k, ["최악순위", "합계_순위"])["station_key"]
        .tolist()
    )
    band_list = sorted(band_keys)
    print(f"\n[ablation] SHAP 상위 {args.ablate_k}개: {top}", flush=True)
    ablation = ablation_table(X, y, tr, va, te, [
        ("전체", []),
        (f"SHAP 상위 {args.ablate_k}개 공정 제거", top),
        (f"절벽 구간 전체 {len(band_list)}개 제거", band_list),
        ("공정파생(전역)만 제거", [GLOBAL]),
        ("공정파생(전역) + 절벽 구간 전체 제거", [GLOBAL] + band_list),
    ])
    ablation.to_csv(args.report_dir / "T_ablation.csv", index=False, encoding="utf-8-sig")

    # ---- 출력 -----------------------------------------------------------
    cols = ["station_key", "입력수", "합계_순위", "평균_순위", "상위포함수_순위",
            "최악순위", "상위포함수", "절벽구간"]
    print("\n[공정 기여도 — 세 방식 모두에서 상위인 순]")
    print(contrib[cols].head(15).to_string(index=False))
    print(f"\n[절벽 구간(상대시각 1.0 초과 4.0 이하) 공정 {len(band_keys)}개의 순위]")
    print(contrib.loc[contrib["절벽구간"], cols].to_string(index=False))
    print("\n[ablation — 시험 구간. 포착 건수 노이즈 표준편차는 약 √건수]")
    print(ablation[["조건", "입력수", "포착건수", "기준선대비_건수", "Lift@1%", "PR_AUC"]].to_string(index=False))
    print(f"\n주의: {CAUTION}")

    (Path("runs") / "explain.json").write_text(
        json.dumps(
            {"표본행": int(len(pick)), "표본불량": int(y[pick].sum()),
             "절벽구간공정": sorted(band_keys), "ablation제거": top,
             "ablation": ablation.to_dict(orient="records"), "주의": CAUTION},
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-16 작성. SHAP 표본은 시험 구간에서만 뽑음(학습 구간에서
#     뽑으면 모델이 외운 패턴의 기여도를 보게 됨)
#   - 측정값별 기여도 표는 만들지 않음. 공정 단위까지만 공개함
####################################################################
