####################################################################
# ingest.py — Bosch 원본 CSV를 parquet로 바꾸는 W0 단계 스크립트
#
# 왜 필요한가
#   원본이 CSV 14GB라서 pandas로 한 번에 읽으면 메모리가 터짐.
#   10만 행씩 끊어 읽어 float32로 줄이고, 열 단위로 저장하는
#   parquet으로 다시 씀. 열 단위 저장이라 나중에 필요한 컬럼만
#   골라 읽을 수 있어서 Week 1 분석이 빨라짐.
#
# 무엇을 검증하는가
#   변환은 "조용히 일부를 잃어버리는" 사고가 나기 쉬운 자리임.
#   그래서 원본 행 수를 따로 세어 변환본과 맞는지 확인하고,
#   안 맞으면 실패로 끝냄. 이게 Week 0 관문임.
#
# 주의
#   이 스크립트는 원본 데이터를 data/raw/ 에서 읽음. data/raw/ 는
#   .gitignore에 있음. Kaggle 대회 데이터는 재배포 금지일 수 있으므로
#   원본이나 원본 행을 저장소에 커밋하지 않음.
####################################################################

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 컬럼명 규칙: L{라인}_S{스테이션}_F{측정번호}, 날짜는 _D{번호}
# 예) L3_S36_F3938  → 라인 3, 스테이션 36, 측정(F) 3938
#     L0_S0_D1     → 라인 0, 스테이션 0,  날짜(D) 1
COLUMN_PATTERN = re.compile(r"^L(\d+)_S(\d+)_([FD])(\d+)$")


####################################################################
# parse_column — 컬럼 이름 하나를 라인/스테이션/종류/번호로 쪼갬
#
# 역할  Bosch가 컬럼 이름 안에 공정 구조를 넣어 둔 것을 꺼내 쓴다.
#       이 프로젝트가 Bosch를 고른 이유가 이 구조라서, 파싱이
#       틀리면 뒤의 모든 공정 단위 분석이 같이 틀어진다.
# 방식  정규식 한 번. 규칙에 안 맞는 이름(Id, Response)은 None을
#       돌려주고, 부르는 쪽에서 걸러 낸다.
####################################################################
def parse_column(name: str) -> dict | None:
    m = COLUMN_PATTERN.match(name)
    if m is None:
        return None
    line, station, kind, idx = m.groups()
    return {
        "column": name,
        "line": int(line),
        "station": int(station),
        "kind": kind,  # F=측정값, D=날짜
        "index": int(idx),
        "station_key": f"L{line}_S{station}",
    }


####################################################################
# count_csv_rows — 원본 CSV의 데이터 행 수를 따로 셈
#
# 역할  변환본과 맞춰 볼 기준값을 만든다. pandas로 세지 않고
#       파일을 바이트로 훑어 줄바꿈만 센다. 메모리를 거의 쓰지 않고,
#       pandas의 파싱 규칙과 독립적인 숫자를 얻는 것이 목적이다.
#       같은 도구로 두 번 세면 같이 틀려도 알아챌 수 없다.
# 방식  16MB씩 읽어 b"\n" 개수를 세고 헤더 한 줄을 뺀다. 마지막 줄에
#       줄바꿈이 없는 파일을 위해 끝 바이트를 확인해 보정한다.
# 한계  필드 안에 줄바꿈이 들어 있는 CSV라면 과다 계수가 된다.
#       Bosch는 숫자만 있는 파일이라 해당되지 않지만, 다른 데이터에
#       재사용할 때는 이 가정을 다시 확인할 것.
####################################################################
def count_csv_rows(path: Path, buffer_size: int = 16 * 1024 * 1024) -> int:
    newlines = 0
    last_byte = b""
    with path.open("rb") as fh:
        while chunk := fh.read(buffer_size):
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        newlines += 1  # 마지막 줄에 줄바꿈이 없는 경우
    return newlines - 1  # 헤더 제외


####################################################################
# convert — CSV 한 개를 parquet 한 개로 바꿈
#
# 역할  끊어 읽어 쓰면서, 스키마가 조각마다 흔들리지 않게 첫 조각의
#       스키마로 고정한다. 조각별로 dtype이 달라지면(어떤 조각은
#       전부 결측이라 object가 되는 등) parquet 쓰기가 중간에 깨진다.
# 방식  Id는 int32, Response는 int8, 나머지 측정·날짜 컬럼은 float32로
#       읽는다. float32로 줄이는 이유는 원본 정밀도가 소수 세 자리
#       수준이어서 float64가 필요 없고, 메모리를 절반으로 줄이기
#       때문이다. 결측이 많아 float가 아니면 담을 수 없다.
# 돌려주는 것  (쓴 행 수, 컬럼 수, 걸린 초)
####################################################################
def convert(src: Path, dst: Path, chunksize: int = 100_000) -> tuple[int, int, float]:
    started = time.time()
    header = pd.read_csv(src, nrows=0)
    columns = list(header.columns)

    dtypes: dict[str, str] = {}
    for name in columns:
        if name == "Id":
            dtypes[name] = "int32"
        elif name == "Response":
            dtypes[name] = "int8"
        else:
            dtypes[name] = "float32"

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None  # 첫 조각의 스키마를 여기 들고 있는다
    rows_written = 0

    try:
        reader = pd.read_csv(src, chunksize=chunksize, dtype=dtypes)
        for i, chunk in enumerate(reader):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(dst, schema, compression="snappy")
            elif table.schema != schema:
                # 첫 조각의 스키마로 맞춘다. 안 맞으면 여기서 터지게 두는 것이
                # 낫다 — 조용히 캐스팅하면 나중에 값이 바뀐 것을 못 찾는다.
                table = table.cast(schema)
            writer.write_table(table)
            rows_written += len(chunk)
            if (i + 1) % 5 == 0:
                print(f"    {rows_written:,}행", flush=True)
    finally:
        if writer is not None:
            writer.close()

    return rows_written, len(columns), time.time() - started


####################################################################
# main — 파일 목록을 돌며 변환하고 행 수를 맞춰 봄
#
# 역할  Week 0 관문을 자동으로 판정한다. 한 파일이라도 행 수가
#       안 맞으면 0이 아닌 코드로 끝내서, 뒤 단계로 그냥 넘어가지
#       못하게 막는다.
####################################################################
def main() -> int:
    ap = argparse.ArgumentParser(description="Bosch 원본 CSV를 parquet로 변환함")
    ap.add_argument("--raw-dir", default="data/raw", type=Path)
    ap.add_argument("--out-dir", default="data/interim", type=Path)
    ap.add_argument("--chunksize", default=100_000, type=int)
    ap.add_argument(
        "--files",
        nargs="*",
        default=["train_numeric.csv", "train_date.csv"],
        help="변환할 파일. 기본은 MVP에서 쓰는 두 개(categorical은 제외)",
    )
    args = ap.parse_args()

    failures: list[str] = []
    for filename in args.files:
        src = args.raw_dir / filename
        if not src.exists():
            print(f"[건너뜀] {src} 가 없음")
            failures.append(f"{filename}: 파일 없음")
            continue

        dst = args.out_dir / (src.stem + ".parquet")
        size_gb = src.stat().st_size / 1024**3
        print(f"\n[{filename}] {size_gb:.2f}GB")

        print("  원본 행 수 세는 중...", flush=True)
        expected = count_csv_rows(src)
        print(f"  원본 {expected:,}행")

        print("  변환 중...", flush=True)
        written, n_cols, elapsed = convert(src, dst, args.chunksize)
        out_gb = dst.stat().st_size / 1024**3

        match = written == expected
        print(
            f"  변환본 {written:,}행 / 컬럼 {n_cols:,}개 / "
            f"{out_gb:.2f}GB / {elapsed:.0f}초"
        )
        print(f"  행 수 일치: {'예' if match else '아니오 ← 관문 실패'}")
        if not match:
            failures.append(f"{filename}: 원본 {expected:,} != 변환본 {written:,}")

    print("\n" + "=" * 60)
    if failures:
        print("Week 0 관문 실패")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Week 0 관문 통과 — 행 수가 모두 일치함")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-09-30 작성. 실제 Bosch 데이터로는 아직 돌려 보지 못했음.
#     Kaggle 대회 규정 동의와 API 토큰이 필요하고 그것은 사용자만
#     할 수 있음. 지금은 tests/test_ingest.py 의 합성 데이터로만
#     검증했음 — 행 수 일치, 컬럼 파싱, dtype, parquet 왕복
#   - 실제 데이터로 돌린 뒤 설계문서 7절의 학습 시간 [추정]을
#     실측값으로 바꿀 것
####################################################################
