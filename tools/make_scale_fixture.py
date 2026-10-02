####################################################################
# make_scale_fixture.py — Bosch와 "같은 모양"의 합성 데이터를 만듦
#
# 왜 필요한가
#   원본은 Kaggle 규정 동의와 API 토큰이 있어야 받을 수 있어서 아직
#   없음. 그런데 모델 성능(PR-AUC·불량 Recall·Lift)은 몰라도
#   **처리 성능(변환 시간·메모리·학습 시간)** 은 데이터 모양만 같으면
#   지금 잴 수 있음. 설계문서에 [추정]으로만 적어 둔 숫자를 [실측]으로
#   바꾸기 위한 것임.
#
# 무엇이 "같은 모양"인가
#   행 1,183,747 / 숫자 컬럼 968 / 날짜 컬럼 1,156 /
#   불량률 0.58% / 결측률 81% / 컬럼명 L{라인}_S{공정}_F|D{번호}
#   (위 수치는 Bosch에 대한 [확인 필요] 값임. 원본을 받으면 맞춰 볼 것)
#
# 이 데이터로 하지 말 것
#   **모델 성능을 재서 보고하지 말 것.** 신호를 내가 심었으므로
#   PR-AUC가 얼마든 아무 뜻이 없음. 여기서 얻을 수 있는 것은
#   "이 장비에서 이 크기를 다루는 데 얼마나 걸리나"뿐임.
####################################################################

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 1_183_747
N_NUMERIC = 968
N_DATE = 1_156
DEFECT_RATE = 0.0058
MISSING_RATE = 0.81
N_LINES = 4
N_STATIONS = 52


####################################################################
# make_column_names — Bosch 규칙을 따르는 컬럼 이름을 만듦
#
# 역할  공정마다 측정값 개수가 다르게 흩뿌린다. 개수가 똑같으면
#       나중에 공정 단위 집계의 교란(측정값 많은 공정이 이김)을
#       재현할 수 없다.
####################################################################
def make_column_names(n_cols: int, kind: str, seed: int = 11) -> list[str]:
    rng = np.random.default_rng(seed)
    stations = [(line, st) for line in range(N_LINES) for st in range(N_STATIONS // N_LINES)]
    # 공정별 컬럼 수를 한쪽으로 치우치게 뽑아 개수 차이를 만든다
    weights = rng.gamma(shape=2.0, scale=1.0, size=len(stations))
    counts = np.maximum(1, np.round(weights / weights.sum() * n_cols).astype(int))
    names: list[str] = []
    idx = 0
    for (line, st), c in zip(stations, counts):
        for _ in range(c):
            names.append(f"L{line}_S{st}_{kind}{idx}")
            idx += 1
            if len(names) == n_cols:
                return names
    while len(names) < n_cols:  # 반올림으로 모자라면 채운다
        line, st = stations[-1]
        names.append(f"L{line}_S{st}_{kind}{idx}")
        idx += 1
    return names[:n_cols]


####################################################################
# generate — CSV 두 개(숫자·날짜)를 조각으로 나눠 써서 메모리를 아낌
#
# 역할  전체를 램에 올리지 않고 쓴다. 1,183,747 × 968 float64는
#       9GB가 넘어서, 생성 단계에서 먼저 터지면 측정 자체를 못 한다.
# 방식  제품 단위로 결측(= 그 공정 미방문)을 공정별로 묶어 만든다.
#       컬럼마다 따로 결측을 뿌리면 "한 공정을 통째로 안 지남"이라는
#       실제 구조가 안 나오고, 그러면 경로 분석 쪽 부하가 달라진다.
####################################################################
def generate(out_dir: Path, n_rows: int, chunk: int = 50_000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    num_cols = make_column_names(N_NUMERIC, "F", seed=11)
    dat_cols = make_column_names(N_DATE, "D", seed=12)

    def station_of(name: str) -> str:
        line, st, _ = name.split("_")
        return f"{line}_{st}"

    num_station = np.array([station_of(c) for c in num_cols])
    dat_station = np.array([station_of(c) for c in dat_cols])
    stations = sorted(set(num_station) | set(dat_station))
    st_index = {s: i for i, s in enumerate(stations)}
    num_st_idx = np.array([st_index[s] for s in num_station])
    dat_st_idx = np.array([st_index[s] for s in dat_station])

    npath = out_dir / "train_numeric.csv"
    dpath = out_dir / "train_date.csv"
    rng = np.random.default_rng(42)
    started = time.time()
    written = 0

    with npath.open("w", newline="") as nfh, dpath.open("w", newline="") as dfh:
        nfh.write("Id," + ",".join(num_cols) + ",Response\n")
        dfh.write("Id," + ",".join(dat_cols) + "\n")

        while written < n_rows:
            m = min(chunk, n_rows - written)
            ids = np.arange(written + 1, written + m + 1, dtype=np.int64)

            # 제품마다 공정 방문 여부. 공정 단위로 끊어야 실제 구조와 닮는다
            visited = rng.random((m, len(stations))) > MISSING_RATE

            num = rng.normal(size=(m, N_NUMERIC)).astype(np.float32)
            num[~visited[:, num_st_idx]] = np.nan

            start = rng.uniform(0, 1700, size=(m, 1)).astype(np.float32)
            dat = start + rng.uniform(0, 6, size=(m, N_DATE)).astype(np.float32)
            dat[~visited[:, dat_st_idx]] = np.nan

            response = (rng.random(m) < DEFECT_RATE).astype(np.int8)

            nblock = pd.DataFrame(num, columns=num_cols)
            nblock.insert(0, "Id", ids)
            nblock["Response"] = response
            nblock.to_csv(nfh, header=False, index=False, lineterminator="\n")

            dblock = pd.DataFrame(dat, columns=dat_cols)
            dblock.insert(0, "Id", ids)
            dblock.to_csv(dfh, header=False, index=False, lineterminator="\n")

            written += m
            del num, dat, nblock, dblock, visited
            if written % 200_000 == 0:
                print(f"    {written:,}행 ({time.time() - started:.0f}초)", flush=True)

    return {
        "행": written,
        "숫자 컬럼": N_NUMERIC,
        "날짜 컬럼": N_DATE,
        "공정": len(stations),
        "생성 초": round(time.time() - started, 1),
        "numeric GB": round(npath.stat().st_size / 1024**3, 2),
        "date GB": round(dpath.stat().st_size / 1024**3, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Bosch와 같은 모양의 합성 데이터 생성")
    ap.add_argument("--out-dir", default="data/raw", type=Path)
    ap.add_argument("--rows", default=N_ROWS, type=int)
    args = ap.parse_args()

    print(f"합성 데이터 생성 — {args.rows:,}행")
    print("주의: 이 데이터로 모델 성능을 재서 보고하지 말 것. 처리 성능 측정 전용임\n")
    info = generate(args.out_dir, args.rows)
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


####################################################################
# 작업 메모
#   - 2026-10-02 작성. 처리 성능 기준선을 재기 위한 것이고,
#     원본을 받으면 같은 측정을 다시 해서 비교할 것
#   - 모양 수치(118만 행·968·1156·0.58%·81%)는 Bosch에 대한
#     [확인 필요] 값임. 원본에서 다르면 이 상수를 고칠 것
####################################################################
