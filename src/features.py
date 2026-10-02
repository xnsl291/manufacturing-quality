####################################################################
# features.py — 제조 관점 입력값을 만들고 시간 순으로 나눔 (Week 2)
#
# 무엇을 만드는가
#   원시 측정값 968개는 그대로 쓰고, 여기서는 **공정 흐름에서 나오는
#   값**만 만든다. 어느 공정을 지났나 / 몇 개를 지났나 / 언제 시작해
#   언제 끝났나 / 측정값이 몇 개 있나 / 어떤 경로로 지나갔나.
#
# 왜 방문 여부를 날짜 쪽에서 만드는가 — Week 1에서 알게 된 것
#   L3_S42, L3_S46 은 **날짜는 기록되지만 측정값이 없는 공정**이다.
#   숫자 컬럼만 보고 방문 여부를 만들면 이 두 공정을 지난 사실을
#   아예 잃는다. 그래서 방문 여부는 날짜 쪽을 기준으로 한다.
#   (숫자 쪽 공정 50개 + 날짜 전용 2개 = 52개)
#
# 누설을 막기 위해 지키는 것
#   경로를 "상위 N개 + 기타"로 묶을 때, 상위 N개를 **학습 구간에서만**
#   고른다. 전체에서 고르면 검증·시험 구간의 분포가 학습 설계에
#   새어 들어온다.
#
# 날짜가 아예 없는 제품
#   582명이 날짜가 전혀 없다. 시간 순으로 줄을 세울 수 없으므로
#   **학습 구간에 넣는다.** 검증·시험에 넣으면 그쪽을 오염시키고,
#   버리면 평가 대상 집단이 달라진다. 수를 기록해 둔다.
####################################################################

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest import parse_column  # noqa: E402

TRAIN_FRAC = 0.60
VALID_FRAC = 0.20  # 나머지가 시험 구간
TOP_PATHS = 20


####################################################################
# column_map — parquet 컬럼 이름을 쪼개 표로 만듦
####################################################################
def column_map(parquet_path: Path) -> pd.DataFrame:
    names = pq.read_schema(parquet_path).names
    return pd.DataFrame([c for c in (parse_column(n) for n in names) if c])


####################################################################
# station_time_matrix — 공정별로 (방문 여부, 그 공정의 최초 시각)을 구함
#
# 역할  방문 여부와 시간 구조가 모두 여기서 나온다. 공정 하나씩 필요한
#       컬럼만 읽어 램을 아낀다.
# 돌려주는 것  (방문 bool DataFrame, 공정 최초시각 float DataFrame)
####################################################################
def station_time_matrix(date_parquet: Path, dmap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    visited: dict[str, np.ndarray] = {}
    first: dict[str, np.ndarray] = {}
    for station_key, group in dmap.groupby("station_key", sort=True):
        block = pq.read_table(date_parquet, columns=group["column"].tolist()).to_pandas()
        mn = block.min(axis=1)
        visited[station_key] = mn.notna().to_numpy()
        first[station_key] = mn.to_numpy(dtype="float32")
        del block
    return pd.DataFrame(visited), pd.DataFrame(first)


####################################################################
# measurement_counts — 제품별 측정값 개수와 결측 개수를 셈
#
# 역할  "측정이 얼마나 촘촘히 됐나"를 입력으로 쓴다. 경로가 짧으면
#       측정값도 적으므로 경로 정보와 겹치지만, 같은 경로 안에서도
#       측정 누락 정도가 다를 수 있어 따로 둔다.
# 방식  숫자 컬럼을 공정 단위로 끊어 읽어 누적한다. 968개를 한꺼번에
#       올리지 않는다.
####################################################################
def measurement_counts(numeric_parquet: Path, nmap: pd.DataFrame) -> pd.DataFrame:
    total = len(nmap)
    present = None
    for _, group in nmap.groupby("station_key", sort=True):
        block = pq.read_table(numeric_parquet, columns=group["column"].tolist()).to_pandas()
        c = block.notna().sum(axis=1).to_numpy(dtype="int32")
        present = c if present is None else present + c
        del block
    return pd.DataFrame(
        {
            "measurement_count": present,
            "missing_count": (total - present).astype("int32"),
            "missing_ratio": (1.0 - present / total).astype("float32"),
        }
    )


####################################################################
# assign_split — 시작 시각 순으로 60/20/20 으로 나눔
#
# 역할  과거로 학습해 이후 생산품을 판정하는 조건을 만든다.
# 방식  순위 기준으로 자른다. 분위수 값으로 자르면 같은 시각에 몰린
#       제품 때문에 구간 크기가 들쭉날쭉해진다.
#       날짜가 없는 제품은 맨 앞(학습)으로 보낸다.
####################################################################
def assign_split(start: pd.Series) -> pd.Series:
    n = len(start)
    # 날짜 없는 제품을 맨 앞에 두기 위해 결측을 -inf 로 취급한다
    key = start.fillna(-np.inf).to_numpy()
    order = np.argsort(key, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    split = np.full(n, "test", dtype=object)
    split[rank < int(n * (TRAIN_FRAC + VALID_FRAC))] = "valid"
    split[rank < int(n * TRAIN_FRAC)] = "train"
    return pd.Series(split, name="split")


####################################################################
# path_ids — 경로를 "학습 구간 상위 N개 + 기타"로 묶음
#
# 역할  제품 종류 대용으로 쓴다. 경로별 불량률 차이가 Week 1에서
#       약 2배까지 확인됐으므로 의미가 있다.
# 방식  방문 여부에 2의 거듭제곱을 곱해 정수 키를 만든다. 상위 N개를
#       **학습 구간에서만** 고른다 — 전체에서 고르면 누설이다.
#       상위에 없는 경로는 모두 0(기타)으로 보낸다.
####################################################################
def path_ids(visited: pd.DataFrame, split: pd.Series, top_n: int = TOP_PATHS) -> pd.DataFrame:
    if visited.shape[1] > 63:
        raise ValueError(f"공정이 {visited.shape[1]}개여서 int64 비트키에 담기지 않음")
    weights = (1 << np.arange(visited.shape[1], dtype=np.int64)).astype(np.int64)
    keys = visited.to_numpy().astype(np.int64) @ weights

    is_train = (split == "train").to_numpy()
    top = pd.Series(keys[is_train]).value_counts().head(top_n).index.to_numpy()
    lookup = {k: i + 1 for i, k in enumerate(top)}  # 0 은 기타
    path_id = np.array([lookup.get(k, 0) for k in keys], dtype=np.int16)
    return pd.DataFrame({"path_key": keys, "path_id": path_id})


####################################################################
# build — 위 조각을 모아 공정 파생 입력값 표를 만듦
#
# 역할  원시 측정값 968개는 손대지 않고, 공정 흐름 입력값만 만들어
#       따로 저장한다. 두 벌을 합쳐 디스크에 또 쓰면 5GB가 중복되므로,
#       합치는 것은 학습할 때 램에서 한다.
####################################################################
def build(numeric_parquet: Path, date_parquet: Path) -> tuple[pd.DataFrame, dict]:
    started = time.time()
    nmap = column_map(numeric_parquet)
    dmap = column_map(date_parquet)

    visited, first = station_time_matrix(date_parquet, dmap)
    start = first.min(axis=1)
    end = first.max(axis=1)

    counts = measurement_counts(numeric_parquet, nmap)
    split = assign_split(start)
    paths = path_ids(visited, split)

    # 방문 플래그 (공정 52개) — 이름을 명확히 해 둔다
    visit_flags = visited.astype("int8").add_prefix("visited_")

    line_of = dmap.drop_duplicates("station_key").set_index("station_key")["line"]
    lines = sorted(line_of.unique())
    line_visited = pd.DataFrame(
        {
            f"visited_line_{ln}": visited[[s for s in visited.columns if line_of[s] == ln]]
            .any(axis=1)
            .astype("int8")
            for ln in lines
        }
    )

    # Id 를 함께 둔다. 숫자 parquet 과 행 순서로만 맞추면, 어느 한쪽을
    # 재생성하는 순간 조용히 어긋난다. 학습할 때 Id 로 검증한다.
    ids = pq.read_table(numeric_parquet, columns=["Id"]).to_pandas()["Id"]

    feats = pd.concat(
        [
            ids,
            visit_flags,
            line_visited,
            pd.DataFrame(
                {
                    "visited_station_count": visited.sum(axis=1).astype("int16"),
                    "visited_line_count": line_visited.sum(axis=1).astype("int8"),
                    "start_time": start.astype("float32"),
                    "end_time": end.astype("float32"),
                    "duration": (end - start).astype("float32"),
                }
            ),
            counts,
            paths[["path_id"]],
        ],
        axis=1,
    )
    feats["split"] = split.to_numpy()

    info = {
        "행": int(len(feats)),
        "공정 파생 입력값 수": int(feats.shape[1] - 2),  # Id, split 제외
        "방문 플래그 공정 수": int(visited.shape[1]),
        "날짜가 없는 제품": int(start.isna().sum()),
        "구간별 행 수": {k: int(v) for k, v in split.value_counts().items()},
        "학습 구간 상위 경로 수": int(paths["path_id"].max()),
        "기타 경로로 묶인 제품": int((paths["path_id"] == 0).sum()),
        "걸린 초": round(time.time() - started, 1),
    }
    return feats, info


def main() -> int:
    ap = argparse.ArgumentParser(description="공정 파생 입력값 생성 + 시간 분할")
    ap.add_argument("--numeric", default="data/interim/train_numeric.parquet", type=Path)
    ap.add_argument("--date", default="data/interim/train_date.parquet", type=Path)
    ap.add_argument("--out", default="data/interim/process_features.parquet", type=Path)
    ap.add_argument("--report-dir", default="reports", type=Path)
    args = ap.parse_args()

    for p in (args.numeric, args.date):
        if not p.exists():
            print(f"[중단] {p} 가 없음. 먼저 src/ingest.py 를 돌릴 것")
            return 1

    feats, info = build(args.numeric, args.date)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)

    for k, v in info.items():
        print(f"  {k}: {v}")

    # 구간별 불량률을 함께 내 둔다 — 분할이 타당한지 바로 보이게
    response = pq.read_table(args.numeric, columns=["Response"]).to_pandas()["Response"]
    bysplit = (
        pd.DataFrame({"split": feats["split"], "response": response.to_numpy()})
        .groupby("split")
        .agg(제품수=("response", "size"), 불량수=("response", "sum"))
    )
    bysplit["불량률"] = bysplit["불량수"] / bysplit["제품수"]
    bysplit = bysplit.reindex(["train", "valid", "test"])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    bysplit.to_csv(args.report_dir / "T_구간별불량률.csv", encoding="utf-8-sig")
    print("\n[구간별 불량률]")
    print(bysplit.to_string())

    (args.report_dir / "features_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{args.out} 저장함")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-02 작성. 실제 데이터로 돌린 결과는 reports/features_info.json
#   - 체류시간·대기시간·재작업 흔적은 아직 안 넣었음(설계 4절 "실험" 항목).
#     기본 입력으로 먼저 성능을 보고, 개선이 필요할 때 붙임
####################################################################
