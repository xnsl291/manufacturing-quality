####################################################################
# diagnose_week2.py — Week 2 Gate 실패 원인을 좁힘
#
# 왜 이 진단을 하는가
#   2026-10-02 첫 학습에서 두 가지가 설계의 실패 조건에 걸렸다.
#     ① 검사율 1%에서 포착률 11.5% — 예고한 20~35%의 절반
#     ② LightGBM(PR-AUC 0.0148) < Logistic(0.0171)
#   설계 5절은 이때 **모델을 키우지 말고 데이터·설정으로 돌아가라**고
#   정해 두었다. 그래서 가설을 하나씩 끊는다.
#
# 가설
#   H1 시간 분할 때문이다. 학습 기간 불량률 0.730% vs 시험 0.407% 로
#      공정 상태가 달라졌으니 학습한 패턴이 뒤 기간에 덜 통한다.
#      → 같은 설정으로 **무작위 분할**과 비교한다. 무작위에서 성능이
#        확 오르면 시간 분할이 원인이고, 그것은 버그가 아니라 결과다.
#   H2 조기 종료가 너무 일찍 끊었다. 검증 구간 불량이 729개뿐이라
#      PR-AUC 가 들쭉날쭉해 조기 종료가 잘못 걸릴 수 있다.
#      → 조기 종료 없이 고정 라운드로 돌려 비교한다.
#   H3 입력값이 부족하다. 설계 4절의 "실험" 항목(공정 체류시간,
#      공정 간 대기시간)을 아직 넣지 않았다.
#      → 이 스크립트에서는 H1·H2만 끊고, H3 는 결과를 보고 판단한다.
#
# 주의
#   무작위 분할 결과는 **비교용**이다. 최종 성능으로 쓰면 안 된다.
#   미래 정보를 쓰는 조건이므로 운영 성능을 과대평가한다.
####################################################################

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import column_map  # noqa: E402
from src.metrics import summary  # noqa: E402
from src.train import LGB_PARAMS  # noqa: E402


def random_split(n: int, seed: int = 42) -> np.ndarray:
    """시간을 무시한 무작위 60/20/20. 비교용임."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = np.full(n, "test", dtype=object)
    split[order[: int(n * 0.8)]] = "valid"
    split[order[: int(n * 0.6)]] = "train"
    return split


def run_one(name, X, y, tr, va, te, params, rounds, early):
    p = dict(params)
    p["metric"] = "average_precision"
    started = time.time()
    cb = [lgb.early_stopping(50, verbose=False)] if early else []
    booster = lgb.train(
        p,
        lgb.Dataset(X.loc[tr], label=y[tr]),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X.loc[va], label=y[va])],
        callbacks=cb,
    )
    s = summary(y[te], booster.predict(X.loc[te]))
    s["학습초"] = round(time.time() - started, 1)
    s["라운드"] = int(booster.best_iteration or rounds)
    print(f"  {name}: PR-AUC {s['PR_AUC']:.4f} / 포착률@1% {s['포착률@1%']:.2%} "
          f"/ Lift {s['Lift@1%']:.1f} / {s['라운드']}라운드 / {s['학습초']}초", flush=True)
    return name, s


def main() -> int:
    feats = pd.read_parquet(ROOT / "data/interim/process_features.parquet")
    label = pq.read_table(ROOT / "data/interim/train_numeric.parquet",
                          columns=["Id", "Response"]).to_pandas()
    assert (feats["Id"].to_numpy() == label["Id"].to_numpy()).all()
    y = label["Response"].to_numpy()
    proc = feats.drop(columns=["Id", "split"])

    nmap = column_map(ROOT / "data/interim/train_numeric.parquet")
    print("원시 측정값 적재 중...", flush=True)
    raw = pq.read_table(ROOT / "data/interim/train_numeric.parquet",
                        columns=nmap["column"].tolist()).to_pandas()
    X = pd.concat([raw, proc], axis=1)
    del raw

    time_split = feats["split"].to_numpy()
    rand_split = random_split(len(y))

    p_w = dict(LGB_PARAMS); p_w["is_unbalance"] = True
    out = {}

    print("\n[H2] 조기 종료 없이 고정 1,000라운드 — 시간 분할", flush=True)
    for nm, s in [run_one("가중치없음_1000라운드", X, y,
                          time_split == "train", time_split == "valid", time_split == "test",
                          LGB_PARAMS, 1000, early=False),
                  run_one("가중치있음_1000라운드", X, y,
                          time_split == "train", time_split == "valid", time_split == "test",
                          p_w, 1000, early=False)]:
        out[f"시간분할_{nm}"] = s

    print("\n[H1] 무작위 분할 (비교용 — 최종 성능으로 쓰지 않음)", flush=True)
    for nm, s in [run_one("가중치없음_조기종료", X, y,
                          rand_split == "train", rand_split == "valid", rand_split == "test",
                          LGB_PARAMS, 1000, early=True),
                  run_one("가중치있음_조기종료", X, y,
                          rand_split == "train", rand_split == "valid", rand_split == "test",
                          p_w, 1000, early=True)]:
        out[f"무작위분할_{nm}"] = s

    df = pd.DataFrame(out).T[["PR_AUC", "포착률@1%", "Lift@1%", "MCC@1%", "라운드", "학습초"]]
    print("\n" + "=" * 70)
    print(df.to_string())

    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports/T_Week2진단.csv", encoding="utf-8-sig")
    (ROOT / "runs/week2_diagnose.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n주의: 무작위 분할은 미래 정보를 쓰는 조건이라 운영 성능을 과대평가함. 비교용임")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
