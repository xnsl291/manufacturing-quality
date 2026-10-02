####################################################################
# where_is_signal.py — 신호가 어디에 있는지 끊어서 확인
#
# 왜 이 실험을 하는가
#   2026-10-02 결과에서 원시 측정값 968개를 쓴 트리가 공정 요약 265개를
#   쓴 선형 모델을 **뚜렷하게 이기지 못했다**(검사율 1%에서 포착 불량
#   111건 vs 114건, 차이 3건으로 0.3σ). 그렇다면 신호 대부분이 개별
#   측정값이 아니라 **공정 경로·결측 구조**에 있을 수 있다.
#
#   입력을 갈라서 각각 돌려 보면 끊을 수 있다.
#     A 공정 파생만 (65개) — 어느 공정을 지났나, 공정시간, 측정 개수
#     B 원시 측정값만 (968개)
#     C 둘 다 (1,033개) — 이미 측정했음
#
#   A 가 C 에 가깝다면 측정값 자체는 거의 기여하지 않는다는 뜻이고,
#   그것은 이 데이터에 대한 중요한 발견이다. 모델을 키울 일이 아니라
#   입력을 다시 생각할 일이 된다.
#
# 부풀림 요인을 합치면 어떻게 되나
#   무작위 분할(9.0배)과 Id 누설(2.22배)을 **동시에** 쓰면 얼마나
#   부풀려지는지도 같이 잰다. 인터넷에 있는 Bosch 결과들이 흔히 둘 다
#   쓰므로, 그 숫자와 내 숫자의 거리를 설명할 수 있어야 한다.
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
from src.train import LGB_PARAMS, id_neighbor_features  # noqa: E402


def fit(name, X, y, tr, va, te, rounds=1000):
    p = dict(LGB_PARAMS); p["metric"] = "average_precision"
    t0 = time.time()
    b = lgb.train(
        p, lgb.Dataset(X.loc[tr], label=y[tr]), num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X.loc[va], label=y[va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    s = summary(y[te], b.predict(X.loc[te]))
    s["입력수"] = int(X.shape[1]); s["라운드"] = int(b.best_iteration or rounds)
    s["학습초"] = round(time.time() - t0, 1)
    print(f"  {name}: 입력 {s['입력수']}개 / PR-AUC {s['PR_AUC']:.4f} / "
          f"포착률@1% {s['포착률@1%']:.2%} / Lift {s['Lift@1%']:.1f} / "
          f"{s['라운드']}라운드 / {s['학습초']}초", flush=True)
    return name, s


def main() -> int:
    feats = pd.read_parquet(ROOT / "data/interim/process_features.parquet")
    label = pq.read_table(ROOT / "data/interim/train_numeric.parquet",
                          columns=["Id", "Response"]).to_pandas()
    assert (feats["Id"].to_numpy() == label["Id"].to_numpy()).all()
    y = label["Response"].to_numpy()
    ids = label["Id"].to_numpy()
    proc = feats.drop(columns=["Id", "split"])
    split = feats["split"].to_numpy()
    tr, va, te = split == "train", split == "valid", split == "test"

    nmap = column_map(ROOT / "data/interim/train_numeric.parquet")
    out = {}

    print("[A] 공정 파생만 — 어느 공정을 지났나 / 공정시간 / 측정 개수", flush=True)
    k, s = fit("A_공정파생만", proc, y, tr, va, te); out[k] = s

    print("\n[B] 원시 측정값만", flush=True)
    raw = pq.read_table(ROOT / "data/interim/train_numeric.parquet",
                        columns=nmap["column"].tolist()).to_pandas()
    k, s = fit("B_원시측정값만", raw, y, tr, va, te); out[k] = s

    print("\n[C] 둘 다 (재확인)", flush=True)
    X = pd.concat([raw, proc], axis=1)
    del raw
    k, s = fit("C_둘다", X, y, tr, va, te); out[k] = s

    print("\n[D] 부풀림 요인을 합치면 — 무작위 분할 + Id 누설", flush=True)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    rsplit = np.full(len(y), "test", dtype=object)
    rsplit[order[: int(len(y) * 0.8)]] = "valid"
    rsplit[order[: int(len(y) * 0.6)]] = "train"
    X_leak = pd.concat([X, id_neighbor_features(ids, y)], axis=1)
    k, s = fit("D_무작위분할+누설", X_leak, y,
               rsplit == "train", rsplit == "valid", rsplit == "test"); out[k] = s
    del X_leak

    df = pd.DataFrame(out).T[["입력수", "PR_AUC", "포착률@1%", "Lift@1%", "MCC@1%", "라운드", "학습초"]]
    print("\n" + "=" * 76)
    print(df.to_string())

    base = out["C_둘다"]["PR_AUC"]
    print(f"\n공정 파생만으로 얻는 PR-AUC 가 전체의 {out['A_공정파생만']['PR_AUC']/base:.0%}")
    print(f"원시 측정값만으로 얻는 PR-AUC 가 전체의 {out['B_원시측정값만']['PR_AUC']/base:.0%}")
    print(f"무작위 분할 + 누설을 둘 다 쓰면 PR-AUC 가 {out['D_무작위분할+누설']['PR_AUC']/base:.1f}배로 부풀려짐")

    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports/T_신호위치.csv", encoding="utf-8-sig")
    (ROOT / "runs/where_is_signal.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n주의: D 는 쓰면 안 되는 조건을 일부러 합친 것임. 최종 성능이 아님")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
