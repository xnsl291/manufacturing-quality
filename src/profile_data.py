####################################################################
# profile_data.py — Week 1 "데이터가 무엇인지 이해하기" 단계의 계산부
#
# 왜 필요한가
#   모델을 만들기 전에 데이터 구조를 먼저 알아야 함. 특히 이 데이터는
#   측정값 이름이 익명화돼 있어서, 알 수 있는 것은 컬럼 이름에 박힌
#   라인·공정 구조와 값이 비어 있는 패턴뿐임.
#
# 가장 중요한 산출물
#   "값이 비어 있다 = 그 공정을 지나지 않았다" 가설을 **증거로** 확인하는 표.
#   제품마다 숫자가 전부 비어 있는 공정과 날짜가 전부 비어 있는 공정이
#   일치하는지 보면 됨. 일치하면 미방문이 맞고, 그러면 결측을 평균으로
#   채우는 것은 틀린 처리가 됨. 오히려 "어느 공정을 지났나"가 새 입력이 됨.
#
# 설계 원칙 — 가정하지 않고 보고함
#   숫자 쪽 공정 집합과 날짜 쪽 공정 집합이 같을 것이라고 가정하지 않음.
#   양쪽을 따로 구해 교집합·차집합을 그대로 보고함. 실제 데이터가
#   예상과 다르면 그 사실이 표에 나타나게 하는 것이 이 단계의 목적임.
#
# 메모리
#   parquet이 열 단위 저장이라 공정 하나씩 필요한 컬럼만 읽음. 968개를
#   한꺼번에 올리지 않아도 돼서, 이 단계는 램을 거의 쓰지 않음.
#   parquet으로 바꾼 이유가 이것임.
####################################################################

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest import parse_column  # noqa: E402


####################################################################
# build_column_map — parquet 파일의 컬럼 이름을 모두 쪼개 표로 만듦
#
# 역할  뒤의 모든 공정 단위 집계가 이 표를 기준으로 돌아간다. 규칙에
#       안 맞는 이름(Id, Response)은 따로 모아 돌려주어, 예상 못 한
#       이름이 섞여 있으면 바로 눈에 띄게 한다.
# 방식  parquet 스키마만 읽는다. 데이터는 읽지 않아 즉시 끝난다.
# 돌려주는 것  (파싱된 컬럼 표, 파싱 안 된 이름 목록)
####################################################################
def build_column_map(parquet_path: Path) -> tuple[pd.DataFrame, list[str]]:
    schema = pq.read_schema(parquet_path)
    parsed, unparsed = [], []
    for name in schema.names:
        got = parse_column(name)
        if got is None:
            unparsed.append(name)
        else:
            parsed.append(got)
    return pd.DataFrame(parsed), unparsed


####################################################################
# station_all_missing — 제품별로 "이 공정의 값이 전부 비었는가"를 구함
#
# 역할  결측=미방문 가설 검증과 경로 분석이 모두 이 행렬에서 나온다.
# 방식  공정 하나씩, 그 공정에 속한 컬럼만 parquet에서 골라 읽어
#       행 방향으로 전부 결측인지 본다. 공정당 메모리는 컬럼 수 × 행 수
#       뿐이라 전체를 올리지 않는다.
# 돌려주는 것  bool DataFrame (행=제품 순서, 열=station_key)
####################################################################
def station_all_missing(parquet_path: Path, column_map: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, np.ndarray] = {}
    for station_key, group in column_map.groupby("station_key", sort=True):
        cols = group["column"].tolist()
        table = pq.read_table(parquet_path, columns=cols)
        block = table.to_pandas()
        out[station_key] = block.isna().all(axis=1).to_numpy()
        del table, block
    return pd.DataFrame(out)


####################################################################
# check_missing_means_unvisited — 결측=미방문 가설을 증거로 확인
#
# 역할  Week 1의 핵심 표를 만든다. 숫자 쪽과 날짜 쪽에서 각각 "전부
#       결측인 공정"을 구해 제품·공정 칸마다 일치하는지 센다.
# 방식  양쪽에 공통으로 있는 공정만 비교한다. 한쪽에만 있는 공정은
#       비교할 수 없으므로 따로 목록으로 돌려준다 — 가정하지 않고
#       보고하기 위함.
#
#       불일치는 두 종류로 나눈다. 뜻이 전혀 다르기 때문이다.
#         숫자만 결측  = 그 공정을 지났지만 측정은 안 했음 (있을 수 있음)
#         날짜만 결측  = 측정값은 있는데 시각이 없음 (이상 징후)
# 돌려주는 것  (요약 dict, 공정별 표)
####################################################################
def check_missing_means_unvisited(
    numeric_missing: pd.DataFrame, date_missing: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    num_stations = set(numeric_missing.columns)
    dat_stations = set(date_missing.columns)
    common = sorted(num_stations & dat_stations)

    rows = []
    for station in common:
        n = numeric_missing[station].to_numpy()
        d = date_missing[station].to_numpy()
        both = int(np.sum(n & d))  # 둘 다 결측 = 미방문으로 해석
        neither = int(np.sum(~n & ~d))  # 둘 다 있음 = 방문·측정
        numeric_only = int(np.sum(n & ~d))  # 지났지만 측정 안 함
        date_only = int(np.sum(~n & d))  # 측정값은 있고 시각이 없음
        total = both + neither + numeric_only + date_only
        rows.append(
            {
                "station_key": station,
                "미방문(양쪽 결측)": both,
                "방문·측정(양쪽 있음)": neither,
                "숫자만 결측": numeric_only,
                "날짜만 결측": date_only,
                "일치율": (both + neither) / total if total else float("nan"),
            }
        )
    per_station = pd.DataFrame(rows)

    agree = int(per_station["미방문(양쪽 결측)"].sum() + per_station["방문·측정(양쪽 있음)"].sum())
    disagree = int(per_station["숫자만 결측"].sum() + per_station["날짜만 결측"].sum())
    summary = {
        "비교한 공정 수": len(common),
        "숫자에만 있는 공정": sorted(num_stations - dat_stations),
        "날짜에만 있는 공정": sorted(dat_stations - num_stations),
        "전체 일치율": agree / (agree + disagree) if (agree + disagree) else float("nan"),
        "숫자만 결측 칸": int(per_station["숫자만 결측"].sum()),
        "날짜만 결측 칸": int(per_station["날짜만 결측"].sum()),
    }
    return summary, per_station


####################################################################
# path_table — 제품이 거쳐 간 공정 조합(경로)별로 묶어 불량률을 봄
#
# 역할  제조 데이터가 단순 표가 아니라는 것을 보이는 증거. 제품 종류에
#       따라 경로가 다르고, 경로별로 품질이 다를 수 있다.
# 방식  방문 여부(bool 행렬)에 2의 거듭제곱을 곱해 제품마다 정수 키
#       하나를 만든다. 행렬곱 한 번이라 118만 행도 즉시 끝난다.
#       행 단위로 문자열을 이어 붙이면 118만 번 파이썬 루프가 돌아
#       몇 분씩 걸리므로 쓰지 않는다. 읽을 수 있는 공정 목록은 상위
#       N개 경로에만 붙인다 — 표에 실을 것만 풀어 쓰면 되기 때문이다.
#       비율만 보면 표본이 적은 경로가 과장되므로 건수를 같이 둔다.
# 한계  공정 수가 63개를 넘으면 int64에 담기지 않으므로 막아 둔다.
#       Bosch는 52개 수준이라 해당되지 않지만, 다른 데이터에 재사용할
#       때 조용히 틀리는 것보다 멈추는 것이 낫다.
####################################################################
def path_table(
    station_missing: pd.DataFrame, response: pd.Series, top_n: int = 20
) -> pd.DataFrame:
    stations = list(station_missing.columns)
    if len(stations) > 63:
        raise ValueError(
            f"공정이 {len(stations)}개여서 int64 비트키에 담기지 않음. "
            "경로 키 방식을 바꿔야 함"
        )

    visited = (~station_missing).to_numpy()
    weights = (1 << np.arange(len(stations), dtype=np.int64)).astype(np.int64)
    keys = visited.astype(np.int64) @ weights

    df = pd.DataFrame({"key": keys, "response": response.to_numpy()})
    grouped = (
        df.groupby("key")
        .agg(제품수=("response", "size"), 불량수=("response", "sum"))
        .reset_index()
        .sort_values("제품수", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    grouped["불량률"] = grouped["불량수"] / grouped["제품수"]

    def decode(key: int) -> str:
        return "|".join(s for i, s in enumerate(stations) if key >> i & 1)

    grouped["path"] = grouped["key"].map(decode)
    grouped["공정수"] = grouped["key"].map(lambda k: int(bin(k).count("1")))
    return grouped[["path", "공정수", "제품수", "불량수", "불량률"]]


####################################################################
# time_structure — 제품별 시작·종료·소요시간과 공정 순서를 구함
#
# 역할  시간 기반 분할이 가능한지 판정하고, 컬럼명 순서가 실제 공정
#       순서와 같은지 확인한다. 둘 다 Week 1 Gate 질문이다.
# 방식  날짜 컬럼 전체에서 행 방향 최소·최대를 취한다. 날짜 값은 절대
#       시각이 아니라 상대 단위라서, 단위를 해석하지 않고 순서만 쓴다.
#       공정 순서는 공정별 날짜 중앙값을 정렬해 추정한다.
####################################################################
def time_structure(
    date_parquet: Path, column_map: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_station_median = {}
    start = None
    end = None
    for station_key, group in column_map.groupby("station_key", sort=True):
        block = pq.read_table(date_parquet, columns=group["column"].tolist()).to_pandas()
        s = block.min(axis=1)
        e = block.max(axis=1)
        start = s if start is None else np.fmin(start, s)
        end = e if end is None else np.fmax(end, e)
        per_station_median[station_key] = float(block.stack().median()) if block.notna().any().any() else float("nan")
        del block

    per_part = pd.DataFrame({"start": start, "end": end})
    per_part["duration"] = per_part["end"] - per_part["start"]

    order = (
        pd.Series(per_station_median, name="날짜_중앙값")
        .rename_axis("station_key")
        .reset_index()
        .sort_values("날짜_중앙값")
        .reset_index(drop=True)
    )
    order["추정_공정순서"] = order.index + 1
    return per_part, order


####################################################################
# leakage_candidates — Gate 질문 8·9에 답하는 수치를 구함
#
# 역할  Id가 시간과 상관이 있는지, 인접 Id 제품의 정답이 서로 상관되는지
#       본다. 둘 다 "쓰면 안 되는 정보"의 후보이고, 특히 두 번째는
#       대회에서 널리 쓰인 누설이라 수치를 먼저 확보해 둔다.
# 방식  Id-시각 상관은 순위 상관(Spearman)으로 본다. 날짜가 상대 단위라
#       값의 크기 자체는 의미가 없고 순서만 의미가 있기 때문이다.
#
#       인접 상관은 "앞 제품이 불량일 때 이 제품도 불량일 확률"을
#       전체 불량률과 비교한다. 배수가 1에 가까우면 상관이 없고,
#       크면 옆 제품의 정답이 정보가 된다는 뜻 — 즉 누설이다.
####################################################################
def leakage_candidates(
    ids: pd.Series, start: pd.Series, response: pd.Series
) -> dict:
    df = pd.DataFrame({"Id": ids.to_numpy(), "start": start.to_numpy(), "response": response.to_numpy()})

    usable = df.dropna(subset=["start"])
    id_time_corr = (
        float(usable["Id"].corr(usable["start"], method="spearman"))
        if len(usable) > 2
        else float("nan")
    )

    by_id = df.sort_values("Id").reset_index(drop=True)
    prev = by_id["response"].shift(1)
    base_rate = float(by_id["response"].mean())
    mask = prev == 1
    cond_rate = float(by_id.loc[mask, "response"].mean()) if mask.any() else float("nan")

    return {
        "Id-시작시각 순위상관": id_time_corr,
        "전체 불량률": base_rate,
        "앞 제품이 불량일 때 불량률": cond_rate,
        "배수": cond_rate / base_rate if base_rate else float("nan"),
        "앞 제품이 불량인 건수": int(mask.sum()),
    }


####################################################################
# split_feasibility — 시간 순으로 자르면 구간별 불량률이 어떻게 되나
#
# 역할  시간 기반 분할을 쓸 수 있는지, 그리고 구간별로 공정 상태가
#       달라졌는지 본다. 구간마다 불량률이 크게 다르면 그것은 버그가
#       아니라 결과이고, 임계값 안정성 분석의 출발점이 된다.
# 방식  시작 시각 순으로 줄 세워 같은 크기로 n등분한다. 분위수 경계가
#       아니라 순위 기준으로 자르는 이유는, 날짜 값이 몰려 있어도
#       구간 크기가 같게 유지되기 때문이다.
####################################################################
def split_feasibility(
    start: pd.Series, response: pd.Series, n_bins: int = 5
) -> pd.DataFrame:
    df = pd.DataFrame({"start": start.to_numpy(), "response": response.to_numpy()})
    df = df.sort_values("start", kind="stable", na_position="last").reset_index(drop=True)
    df["구간"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    out = (
        df.groupby("구간")
        .agg(
            제품수=("response", "size"),
            불량수=("response", "sum"),
            시작_최소=("start", "min"),
            시작_최대=("start", "max"),
        )
        .reset_index()
    )
    out["불량률"] = out["불량수"] / out["제품수"]
    return out


####################################################################
# main — 위 계산을 순서대로 돌려 표를 reports/ 아래에 저장
####################################################################
def main() -> int:
    ap = argparse.ArgumentParser(description="Week 1 데이터 구조 분석")
    ap.add_argument("--numeric", default="data/interim/train_numeric.parquet", type=Path)
    ap.add_argument("--date", default="data/interim/train_date.parquet", type=Path)
    ap.add_argument("--out-dir", default="reports", type=Path)
    args = ap.parse_args()

    for p in (args.numeric, args.date):
        if not p.exists():
            print(f"[중단] {p} 가 없음. 먼저 src/ingest.py 를 돌릴 것")
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    num_map, num_unparsed = build_column_map(args.numeric)
    dat_map, dat_unparsed = build_column_map(args.date)
    print(f"숫자 컬럼 {len(num_map):,}개 / 규칙 밖 이름 {num_unparsed}")
    print(f"날짜 컬럼 {len(dat_map):,}개 / 규칙 밖 이름 {dat_unparsed}")

    structure = (
        num_map.groupby(["line", "station", "station_key"])
        .size()
        .reset_index(name="측정값수")
        .sort_values(["line", "station"])
    )
    structure.to_csv(args.out_dir / "T2_구조.csv", index=False, encoding="utf-8-sig")
    print(f"라인 {structure['line'].nunique()}개 / 공정 {len(structure)}개")

    print("\n공정별 결측 패턴 계산 중...")
    num_missing = station_all_missing(args.numeric, num_map)
    dat_missing = station_all_missing(args.date, dat_map)

    summary, per_station = check_missing_means_unvisited(num_missing, dat_missing)
    per_station.to_csv(args.out_dir / "T_결측=미방문_검증.csv", index=False, encoding="utf-8-sig")
    print("\n[결측 = 미방문 가설]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    response = pq.read_table(args.numeric, columns=["Response"]).to_pandas()["Response"]
    ids = pq.read_table(args.numeric, columns=["Id"]).to_pandas()["Id"]

    paths = path_table(num_missing, response)
    paths.to_csv(args.out_dir / "T3_경로.csv", index=False, encoding="utf-8-sig")
    print(f"\n상위 {len(paths)}개 경로가 전체의 "
          f"{paths['제품수'].sum() / len(response):.1%}를 덮음")

    per_part, order = time_structure(args.date, dat_map)
    order.to_csv(args.out_dir / "T_공정순서추정.csv", index=False, encoding="utf-8-sig")

    leak = leakage_candidates(ids, per_part["start"], response)
    print("\n[누설 후보]")
    for k, v in leak.items():
        print(f"  {k}: {v}")

    split = split_feasibility(per_part["start"], response)
    split.to_csv(args.out_dir / "T_분할가능성.csv", index=False, encoding="utf-8-sig")
    print("\n[시간 분할 구간별 불량률]")
    print(split[["구간", "제품수", "불량수", "불량률"]].to_string(index=False))

    print(f"\n표를 {args.out_dir}/ 아래에 저장함")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-01 작성. 실제 Bosch 데이터로는 아직 돌려 보지 못했음.
#     Kaggle 규정 동의와 API 토큰이 필요하고 그것은 사용자만 할 수 있음.
#     지금은 tests/test_profile_data.py 의 합성 데이터로만 검증했음 —
#     정답을 알고 만든 데이터에서 계산이 그 값을 되찾는지 확인
#   - 그림(F1~F5)은 아직 안 만듦. 실제 분포를 보기 전에 구간 폭이나
#     축을 정하면 다시 그리게 되므로, 표를 먼저 내고 그림은 데이터를
#     받은 뒤에 붙임
####################################################################
