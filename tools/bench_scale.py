####################################################################
# bench_scale.py — Bosch 크기에서 처리 성능을 실제로 잼
#
# 왜 필요한가
#   설계문서에 [추정]으로만 적어 둔 숫자가 셋 있음 — 변환본 크기,
#   변환 시간, 학습 시간. 원본이 없어 못 쟀는데, 모양이 같은 합성
#   데이터로는 **처리 성능**만은 지금 잴 수 있음.
#
# 재는 것
#   단계별 걸린 시간, 프로세스 최대 메모리(RSS), 산출물 크기
#
# 재지 않는 것 — 중요
#   **모델 성능(PR-AUC·불량 Recall·Lift)은 재지 않음.** 신호를 내가
#   심었으므로 어떤 값이 나와도 뜻이 없음. 학습은 "이 크기를 이 장비가
#   몇 분에 도는가"를 보려고만 돌림.
####################################################################

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


####################################################################
# run_stage — 자식 프로세스로 한 단계를 돌리며 최대 메모리를 지켜봄
#
# 역할  단계마다 따로 프로세스를 띄워 메모리가 단계 간에 섞이지 않게
#       한다. 한 프로세스에서 다 돌리면 앞 단계가 잡아 둔 메모리 때문에
#       뒤 단계의 최대치를 잘못 읽는다.
# 방식  0.2초마다 자식과 그 자손의 RSS 합을 본다. 자식이 numpy로 잡은
#       메모리는 파이썬 추적(tracemalloc)에 안 잡히므로 OS가 보는
#       RSS를 쓴다.
####################################################################
def run_stage(name: str, args: list[str]) -> dict:
    print(f"\n[{name}] 시작", flush=True)
    started = time.time()
    proc = subprocess.Popen(
        [str(PY), *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    peak = {"rss": 0}

    def watch() -> None:
        try:
            p = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            return
        while proc.poll() is None:
            try:
                total = p.memory_info().rss
                for c in p.children(recursive=True):
                    total += c.memory_info().rss
                peak["rss"] = max(peak["rss"], total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.2)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line.rstrip())
        if len(tail) > 12:
            tail.pop(0)
    code = proc.wait()
    t.join(timeout=2)
    elapsed = time.time() - started

    for line in tail:
        print(f"    {line}")
    print(f"[{name}] {elapsed:.0f}초 / 최대 메모리 {peak['rss'] / 1024**3:.2f}GB / 종료코드 {code}")
    return {
        "단계": name,
        "초": round(elapsed, 1),
        "분": round(elapsed / 60, 1),
        "최대 메모리 GB": round(peak["rss"] / 1024**3, 2),
        "종료코드": code,
    }


####################################################################
# train_probe — LightGBM 학습 1회가 몇 분인지만 봄
#
# 역할  설계문서 7절의 "학습 1회 2~8분 [추정]"을 실측으로 바꾼다.
# 방식  설계에 적어 둔 고정 파라미터를 그대로 쓴다. 조기 종료를 켜되
#       라운드 수가 들쭉날쭉하면 시간 비교가 안 되므로, 고정 라운드로
#       한 번 돌려 "트리 1,000개를 세우는 데 몇 분"을 재는 쪽을 택한다.
#       **성능 지표는 찍지 않는다** — 합성 신호라 뜻이 없다.
####################################################################
TRAIN_PROBE = r'''
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

src = Path("data/_bench/interim/train_numeric.parquet")
t0 = time.time()
df = pd.read_parquet(src)
load_s = time.time() - t0
y = df.pop("Response").to_numpy()
df.pop("Id")
X = df
print(f"적재 {load_s:.0f}초 / 모양 {X.shape} / 램상 {X.memory_usage(deep=True).sum()/1024**3:.2f}GB")

t0 = time.time()
ds = lgb.Dataset(X, label=y, free_raw_data=True)
params = dict(objective="binary", num_leaves=63, learning_rate=0.05,
              min_child_samples=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=5, verbose=-1, seed=42,
              num_threads=0)
booster = lgb.train(params, ds, num_boost_round=1000)
train_s = time.time() - t0
print(f"학습 1,000라운드 {train_s:.0f}초 ({train_s/60:.1f}분)")
print(json.dumps({"적재초": round(load_s,1), "학습초": round(train_s,1),
                  "행": int(X.shape[0]), "열": int(X.shape[1])}, ensure_ascii=False))
'''


def main() -> int:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1_183_747
    results = []

    # 합성 데이터는 data/_bench/ 아래에만 둔다. data/raw 에 쓰면 나중에
    # 실제 데이터와 섞이거나, 합성 데이터를 실제인 줄 알고 분석하게 된다.
    raw = "data/_bench/raw"
    interim = "data/_bench/interim"
    results.append(
        run_stage(
            "1. 합성 데이터 생성",
            ["tools/make_scale_fixture.py", "--rows", str(rows), "--out-dir", raw],
        )
    )
    results.append(
        run_stage("2. CSV → parquet 변환", ["src/ingest.py", "--raw-dir", raw, "--out-dir", interim])
    )
    results.append(
        run_stage(
            "3. Week 1 분석",
            [
                "src/profile_data.py",
                "--numeric", f"{interim}/train_numeric.parquet",
                "--date", f"{interim}/train_date.parquet",
                "--out-dir", "reports/_bench",
            ],
        )
    )

    probe = ROOT / "tools" / "_train_probe.py"
    probe.write_text(TRAIN_PROBE, encoding="utf-8")
    results.append(run_stage("4. LightGBM 학습 1,000라운드", [str(probe)]))
    probe.unlink(missing_ok=True)

    print("\n" + "=" * 64)
    print("처리 성능 측정 결과 (합성 데이터, Bosch와 같은 모양)")
    print("=" * 64)
    table = pd.DataFrame(results) if (pd := __import__("pandas")) else None
    print(table.to_string(index=False))

    sizes = {}
    for p in sorted((ROOT / "data").rglob("*")):
        if p.is_file():
            sizes[str(p.relative_to(ROOT))] = round(p.stat().st_size / 1024**3, 2)
    print("\n파일 크기 (GB)")
    for k, v in sizes.items():
        print(f"  {k}: {v}")

    out = ROOT / "runs" / "scale_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"행": rows, "단계": results, "파일GB": sizes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n{out} 에 저장함")
    print("\n주의: 여기 숫자는 처리 성능임. 모델 성능(PR-AUC·Recall·Lift)이 아님")
    return 0 if all(r["종료코드"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
