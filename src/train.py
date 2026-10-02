####################################################################
# train.py — 모델 셋을 학습해 비교하고, Id 누설을 일부러 재현함 (Week 2)
#
# 모델을 세 개만 두는 이유
#   모델 비교 대회로 만들지 않는다. 필요한 것은 세 가지뿐이다.
#     Dummy     불량률 0.58% 환경에서 정확도 99.4%가 왜 무의미한지
#     Logistic  선형 기준선. 트리가 실제로 얼마나 개선하는지 재는 자
#     LightGBM  본 모델
#
# Id 누설을 일부러 재현하는 이유
#   Week 1에서 "앞 제품(Id순)이 불량일 때 불량률이 전체의 9.38배"임을
#   확인했다. Id 는 시작 시각과 상관이 거의 없는데도(순위상관 0.002)
#   그렇다. 즉 Id 순서 자체에 정답 정보가 들어 있다.
#   운영에서는 옆 제품의 검사 결과를 미리 알 수 없으므로 쓰면 안 되는
#   정보다. 그래서 **일부러 넣어 성능이 얼마나 부풀려지는지 수치로
#   보인 뒤 배제**한다. "점검했다"와 "실재함을 보이고 버렸다"의 차이다.
#
# 불균형 대응을 먼저 하지 않는 이유
#   최종 임계값을 확률 절대값이 아니라 검사 물량 분위수로 정하므로
#   필요한 것은 순위다. 가중치는 순위를 크게 바꾸지 못하면서 확률만
#   왜곡한다. 가중치 유무는 실험 한 번으로 재서 표에 남긴다.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import column_map  # noqa: E402
from src.metrics import cost_curve, summary, threshold_stability, threshold_table  # noqa: E402

LGB_PARAMS = dict(
    objective="binary",
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=100,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
    seed=42,
    num_threads=0,
)


####################################################################
# station_summary — 공정마다 측정값의 평균·표준편차를 구함
#
# 역할  Logistic 용 축약 입력값. 선형 모델은 968개 전부에 결측 대치가
#       필요해 그대로 쓰기 어렵고, 공정 단위로 묶으면 100개로 줄어든다.
# 방식  공정 하나씩 읽어 행 방향 평균·표준편차를 낸다. 그 공정을 지나지
#       않은 제품은 결측으로 남고, 뒤에서 중앙값으로 채우며 결측 플래그를
#       따로 둔다(결측 자체가 정보라는 Week 1 결론 때문).
####################################################################
def station_summary(numeric_parquet: Path, nmap: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for station_key, group in nmap.groupby("station_key", sort=True):
        block = pq.read_table(numeric_parquet, columns=group["column"].tolist()).to_pandas()
        out[f"{station_key}_mean"] = block.mean(axis=1).to_numpy(dtype="float32")
        out[f"{station_key}_std"] = block.std(axis=1).to_numpy(dtype="float32")
        del block
    return pd.DataFrame(out)


####################################################################
# id_neighbor_features — Id 순서 누설 입력값을 만듦 (실험용, 최종 배제)
#
# 역할  Id 로 정렬했을 때 앞·뒤 제품의 **정답**과 Id 간격을 입력으로
#       만든다. 운영에서는 얻을 수 없는 정보다.
# 주의  이 함수가 만드는 것은 **누설이다.** 최종 모델에 넣지 않는다.
#       부풀려진 성능을 수치로 보이기 위한 실험에만 쓴다.
####################################################################
def id_neighbor_features(ids: np.ndarray, response: np.ndarray) -> pd.DataFrame:
    order = np.argsort(ids, kind="stable")
    y_sorted = response[order]
    id_sorted = ids[order]

    prev_y = np.concatenate([[np.nan], y_sorted[:-1]])
    next_y = np.concatenate([y_sorted[1:], [np.nan]])
    prev_gap = np.concatenate([[np.nan], np.diff(id_sorted)]).astype("float32")
    next_gap = np.concatenate([np.diff(id_sorted), [np.nan]]).astype("float32")

    out = pd.DataFrame(
        {
            "leak_prev_response": prev_y.astype("float32"),
            "leak_next_response": next_y.astype("float32"),
            "leak_prev_id_gap": prev_gap,
            "leak_next_id_gap": next_gap,
        }
    )
    # 원래 행 순서로 되돌린다
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return out.iloc[inverse].reset_index(drop=True)


####################################################################
# fit_lgb — LightGBM 한 번 학습. 조기 종료는 검증 PR-AUC 기준
####################################################################
def fit_lgb(X_tr, y_tr, X_va, y_va, params=None, rounds=1000, patience=50):
    p = dict(LGB_PARAMS if params is None else params)
    p["metric"] = "average_precision"
    started = time.time()
    booster = lgb.train(
        p,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return booster, round(time.time() - started, 1)


####################################################################
# main — 모델 셋 + 가중치 비교 + 누설 실험을 돌려 표를 남김
####################################################################
def main() -> int:
    ap = argparse.ArgumentParser(description="Week 2 모델 학습·비교·누설 실험")
    ap.add_argument("--numeric", default="data/interim/train_numeric.parquet", type=Path)
    ap.add_argument("--features", default="data/interim/process_features.parquet", type=Path)
    ap.add_argument("--report-dir", default="reports", type=Path)
    ap.add_argument("--run-dir", default="runs/week2", type=Path)
    ap.add_argument("--rounds", default=1000, type=int)
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    print("입력 적재 중...", flush=True)
    feats = pd.read_parquet(args.features)
    label = pq.read_table(args.numeric, columns=["Id", "Response"]).to_pandas()

    # 행 순서가 어긋나지 않았는지 Id 로 확인한다 — 조용히 틀리는 자리다
    assert len(feats) == len(label), "행 수가 다름"
    assert (feats["Id"].to_numpy() == label["Id"].to_numpy()).all(), "Id 순서가 어긋남"

    split = feats["split"].to_numpy()
    y = label["Response"].to_numpy()
    ids = label["Id"].to_numpy()
    proc = feats.drop(columns=["Id", "split"])

    nmap = column_map(args.numeric)
    tr, va, te = split == "train", split == "valid", split == "test"
    print(f"  학습 {tr.sum():,} / 검증 {va.sum():,} / 시험 {te.sum():,}", flush=True)

    results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    # ---- Model 0 Dummy -------------------------------------------------
    # 가장 흔한 쪽(양품)으로만 답하는 모델. "전부 양품"이라는 **단정**이므로
    # 점수에 임계값을 매기는 틀이 적용되지 않는다.
    #
    # 처음에는 점수를 전부 0으로 두고 summary() 를 그대로 썼는데, 전부 동점이라
    # 분위수 임계값이 0이 되어 **전부 알람**으로 집계됐다(포착률 100%, Lift 1.0).
    # 숫자 자체는 틀리지 않지만 Dummy 를 두는 목적을 못 보여준다. 목적은
    # "정확도는 99.6%인데 불량은 하나도 못 잡는다"를 보이는 것이다.
    print("\n[Model 0] Dummy — 전부 양품으로 단정", flush=True)
    results["0_Dummy"] = {
        "PR_AUC": float(y[te].mean()),  # 정보가 없는 모델의 PR-AUC 하한 = 불량률
        "포착률@1%": 0.0,
        "정밀도@1%": float("nan"),  # 알람이 0건이라 정의되지 않음
        "Lift@1%": 0.0,
        "실제검사율@1%": 0.0,
        "MCC@1%": float("nan"),  # 한쪽만 예측
        "정확도(참고용)": float((y[te] == 0).mean()),
        "불량률": float(y[te].mean()),
    }

    # ---- Model 1 Logistic (축약 입력) ---------------------------------
    print("[Model 1] Logistic — 공정 요약 입력 만드는 중...", flush=True)
    ssum = station_summary(args.numeric, nmap)
    X_red = pd.concat([ssum, proc], axis=1)
    # 결측 플래그를 먼저 만들고 나서 채운다 (결측 자체가 정보)
    na_cols = [c for c in ssum.columns]
    flags = ssum[na_cols].isna().astype("int8").add_suffix("_isna")
    X_red = pd.concat([X_red, flags], axis=1)
    med = X_red.loc[tr].median(numeric_only=True)  # 학습 구간에서만 구함
    X_red = X_red.fillna(med).fillna(0.0).astype("float32")
    print(f"  축약 입력 {X_red.shape[1]}개", flush=True)

    started = time.time()
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
    )
    logit.fit(X_red.loc[tr], y[tr])
    timings["1_Logistic"] = round(time.time() - started, 1)
    s_va_lr = logit.predict_proba(X_red.loc[va])[:, 1]
    s_te_lr = logit.predict_proba(X_red.loc[te])[:, 1]
    results["1_Logistic"] = summary(y[te], s_te_lr)
    print(f"  {timings['1_Logistic']}초", flush=True)

    del X_red, ssum, flags

    # ---- Model 2 LightGBM (원시 968 + 공정 파생) -----------------------
    print("[Model 2] LightGBM — 원시 측정값 적재 중...", flush=True)
    raw = pq.read_table(args.numeric, columns=nmap["column"].tolist()).to_pandas()
    X = pd.concat([raw, proc], axis=1)
    del raw
    print(f"  입력 {X.shape[1]}개 / 램 {X.memory_usage(deep=True).sum()/1024**3:.2f}GB", flush=True)

    booster, secs = fit_lgb(X.loc[tr], y[tr], X.loc[va], y[va], rounds=args.rounds)
    timings["2_LightGBM"] = secs
    s_va = booster.predict(X.loc[va])
    s_te = booster.predict(X.loc[te])
    results["2_LightGBM"] = summary(y[te], s_te)
    print(f"  {secs}초 / 최적 라운드 {booster.best_iteration}", flush=True)

    # ---- 가중치 유무 비교 (실험 1회) ------------------------------------
    print("[실험] is_unbalance=True 비교", flush=True)
    p_w = dict(LGB_PARAMS); p_w["is_unbalance"] = True
    bw, secs_w = fit_lgb(X.loc[tr], y[tr], X.loc[va], y[va], params=p_w, rounds=args.rounds)
    results["2b_LightGBM_가중치"] = summary(y[te], bw.predict(X.loc[te]))
    timings["2b_LightGBM_가중치"] = secs_w
    del bw

    # ---- 누설 실험: Id 인접 정답을 일부러 넣음 --------------------------
    print("[실험] Id 누설 재현 — 일부러 넣어 본다", flush=True)
    leak = id_neighbor_features(ids, y)
    X_leak = pd.concat([X, leak], axis=1)
    bl, secs_l = fit_lgb(X_leak.loc[tr], y[tr], X_leak.loc[va], y[va], rounds=args.rounds)
    results["X_누설포함(배제대상)"] = summary(y[te], bl.predict(X_leak.loc[te]))
    timings["X_누설포함(배제대상)"] = secs_l
    leak_gain = pd.DataFrame(
        [
            {"입력": "Id 누설 제외 (최종 모델)", **results["2_LightGBM"]},
            {"입력": "Id 누설 포함 (배제 대상)", **results["X_누설포함(배제대상)"]},
        ]
    )
    leak_gain.to_csv(args.report_dir / "T_누설실험.csv", index=False, encoding="utf-8-sig")
    del X_leak, bl, leak

    # ---- 표 저장 --------------------------------------------------------
    comp = pd.DataFrame(results).T
    comp.insert(0, "학습초", pd.Series(timings))
    comp.to_csv(args.report_dir / "T_모델비교.csv", encoding="utf-8-sig")
    print("\n[모델 비교 — 시험 구간]")
    print(comp.to_string())

    table = threshold_table(y[te], s_te)
    table.to_csv(args.report_dir / "T_검사물량대비포착률.csv", index=False, encoding="utf-8-sig")
    print("\n[검사 물량 대비 포착률 — LightGBM, 시험 구간] ★ 대표 산출물")
    print(table.to_string(index=False))

    stab = threshold_stability(y[va], s_va, y[te], s_te)
    stab.to_csv(args.report_dir / "T_임계값안정성.csv", index=False, encoding="utf-8-sig")
    print("\n[임계값 안정성 — 검증에서 정한 임계값을 시험에 적용]")
    print(stab.to_string(index=False))

    cost = cost_curve(table, defect_rate=float(y[te].mean()))
    cost.to_csv(args.report_dir / "T_비용비곡선.csv", index=False, encoding="utf-8-sig")
    print("\n[비용비 r 별 최적 검사율]")
    print(cost.to_string(index=False))

    booster.save_model(str(args.run_dir / "model_lgb.txt"))
    np.save(args.run_dir / "score_valid.npy", s_va)
    np.save(args.run_dir / "score_test.npy", s_te)
    np.save(args.run_dir / "score_test_logistic.npy", s_te_lr)
    (args.run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "구간": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
                "구간별_불량률": {
                    "train": float(y[tr].mean()),
                    "valid": float(y[va].mean()),
                    "test": float(y[te].mean()),
                },
                "모델": results,
                "학습초": timings,
                "최적라운드": int(booster.best_iteration),
                "전체초": round(time.time() - t_all, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n전체 {time.time() - t_all:.0f}초. 표는 {args.report_dir}/, 모델은 {args.run_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-02 작성. 누설 실험 결과는 reports/T_누설실험.csv
#   - 최종 모델에는 leak_* 입력이 들어가지 않음. 들어간 것은 배제 대상
#     실험 전용임
####################################################################
