####################################################################
# build_artifacts.py — 그림·화면이 읽을 파일을 실제 데이터에서 만듦 (Week 5)
#
# 왜 따로 두나
#   그림(figures.py)과 화면(app/)은 원본을 읽지 않는다. 원본은 15GB라
#   화면이 뜰 때마다 읽을 수 없고, 무엇보다 원본은 공개 저장소에 없다.
#   그래서 원본이 필요한 계산을 여기 한 곳에 모으고, 결과를 두 종류로
#   나눠 쓴다.
#
#   1) 집계표 reports/T_*.csv — 공정·경로·구간 단위라 커밋함
#        T_포착률곡선.csv        검사율 0.1%~20% 촘촘한 곡선 (대표 그림용)
#        T_경로x공정_방문비율.csv 상위 경로 20개 + 기타 × 공정 52개
#        T_시작시각_분포.csv      시작 시각 구간별 제품 수·불량 수
#   2) 제품 단위 파일 artifacts/scored_parts.parquet — 커밋하지 않음
#        시험 구간 제품마다 위험도·실제 결과·공정 흐름 값. Risk Monitor
#        화면만 쓰고, 로컬에서만 존재한다 (대회 규정의 재배포 금지)
#
# 결측 히트맵을 제품 단위로 그리지 않는 이유
#   설계에는 "제품 2,000대 × 공정 52개"였는데, 그건 실제 제품 행을 그림
#   으로 옮기는 것이다. "결측이 무작위인가 경로인가"는 경로 단위 방문
#   비율로도 똑같이 답할 수 있어서 그쪽으로 바꿈 (2026-10-19 결정)
####################################################################

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import threshold_table  # noqa: E402

# 0.1% 부터 20% 까지. 앞쪽이 결론이 나는 자리라 로그 간격으로 촘촘하게 둔다.
# 대표 표의 검사율(0.5·1·3·5·10%)을 반드시 포함한다 — 빠지면 그림에 찍히는
# 1% 점이 1.03% 가 되어 README 의 헤드라인 숫자(11.5배)와 어긋난다
CURVE_RATES = tuple(sorted(set(np.round(np.geomspace(0.001, 0.20, 60), 5))
                           | {0.005, 0.01, 0.03, 0.05, 0.10}))
START_BINS = 60


####################################################################
# capture_curve — 모델별 검사율 ↔ 포착률 곡선을 한 표로 만듦
#
# 역할  대표 그림의 원본 표. train.py 의 6점 표는 점이 너무 적어 곡선
#       모양(어디서 꺾이는지)이 안 보인다.
# 방식  threshold_table 을 그대로 다시 쓴다. 계산을 새로 짜면 대표 표와
#       그림의 숫자가 어긋날 수 있다. 전수 검사 행은 곡선에서 뺀다.
####################################################################
def capture_curve(y_true: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    parts = []
    for name, score in scores.items():
        t = threshold_table(y_true, score, CURVE_RATES)
        t = t[t["목표_검사율"] < 1.0]
        t.insert(0, "모델", name)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


####################################################################
# path_station_visit — 경로 묶음별로 각 공정을 거친 비율
#
# 역할  F2(결측 구조) 그림의 원본 표. 상위 경로는 정의상 0/1 이고,
#       "기타" 묶음에서만 중간값이 나온다 — 결측이 무작위가 아니라
#       경로로 정해진다는 것을 그대로 보여 준다.
####################################################################
def path_station_visit(feats: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    vcols = [c for c in feats.columns if c.startswith("visited_L")]
    g = feats.groupby("path_id")
    out = g[vcols].mean()
    out.columns = [c.removeprefix("visited_") for c in vcols]
    out.insert(0, "불량률", pd.Series(y, index=feats.index).groupby(feats["path_id"]).mean())
    out.insert(0, "제품수", g.size())
    out.index = out.index.map(lambda i: "기타(희귀 경로)" if i == 0 else f"경로 {i:02d}")
    return out.sort_index()


####################################################################
# start_time_hist — 시작 시각 구간별 제품 수·불량 수
#
# 역할  F4 그림의 원본 표. 시간 분할 경계(60%·80%)를 같이 적어 두어
#       그림에서 학습/검증/시험 구간을 표시할 수 있게 한다.
# 방식  날짜가 없는 제품은 셀 수 없으므로 뺀다. 뺀 수는 main 이 출력한다
#       (2026-10-19 실측 582대).
####################################################################
def start_time_hist(feats: pd.DataFrame, y: np.ndarray, bins: int = START_BINS) -> pd.DataFrame:
    has = feats["start_time"].notna().to_numpy()
    st = feats["start_time"].to_numpy()[has]
    edges = np.linspace(st.min(), st.max(), bins + 1)
    idx = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, bins - 1)
    count = np.bincount(idx, minlength=bins)
    defect = np.bincount(idx, weights=y[has], minlength=bins).astype(int)
    split = feats["split"].to_numpy()[has]
    first_valid = st[split == "valid"].min()
    first_test = st[split == "test"].min()
    return pd.DataFrame(
        {
            "구간_시작": edges[:-1],
            "구간_끝": edges[1:],
            "제품수": count,
            "불량수": defect,
            "불량률": np.where(count > 0, defect / np.maximum(count, 1), np.nan),
            "검증_시작": first_valid,
            "시험_시작": first_test,
        }
    )


####################################################################
# scored_parts — 화면용 제품 단위 표 (커밋하지 않음)
####################################################################
def scored_parts(feats: pd.DataFrame, y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    te = (feats["split"] == "test").to_numpy()
    cols = ["Id", "start_time", "duration", "visited_station_count",
            "visited_line_count", "measurement_count", "path_id"]
    out = feats.loc[te, cols].reset_index(drop=True)
    out["위험도"] = score
    out["실제_불량"] = y[te].astype("int8")
    vcols = [c for c in feats.columns if c.startswith("visited_L")]
    visited = feats.loc[te, vcols].to_numpy().astype(bool)
    names = np.array([c.removeprefix("visited_") for c in vcols])
    out["방문_공정"] = ["|".join(names[row]) for row in visited]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="그림·화면용 집계표와 로컬 산출물 생성")
    ap.add_argument("--numeric", default="data/interim/train_numeric.parquet", type=Path)
    ap.add_argument("--features", default="data/interim/process_features.parquet", type=Path)
    ap.add_argument("--run-dir", default="runs/week2", type=Path)
    ap.add_argument("--report-dir", default="reports", type=Path)
    ap.add_argument("--artifact-dir", default="artifacts", type=Path)
    args = ap.parse_args()

    need = [args.features, args.numeric, args.run_dir / "score_test.npy",
            args.run_dir / "score_test_logistic.npy"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        print("[중단] 다음 파일이 없음. features.py → train.py 순서로 먼저 돌릴 것")
        for p in missing:
            print("  -", p)
        return 1

    feats = pd.read_parquet(args.features)
    label = pq.read_table(args.numeric, columns=["Id", "Response"]).to_pandas()
    assert (feats["Id"].to_numpy() == label["Id"].to_numpy()).all(), "Id 순서가 어긋남"
    y = label["Response"].to_numpy()
    te = (feats["split"] == "test").to_numpy()

    s_lgb = np.load(args.run_dir / "score_test.npy")
    s_lr = np.load(args.run_dir / "score_test_logistic.npy")
    assert len(s_lgb) == te.sum() == len(s_lr), "점수 길이가 시험 구간과 다름"

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    curve = capture_curve(y[te], {"LightGBM": s_lgb, "Logistic": s_lr})
    curve.to_csv(args.report_dir / "T_포착률곡선.csv", index=False, encoding="utf-8-sig")
    visit = path_station_visit(feats, y)
    visit.to_csv(args.report_dir / "T_경로x공정_방문비율.csv", encoding="utf-8-sig")
    hist = start_time_hist(feats, y)
    hist.to_csv(args.report_dir / "T_시작시각_분포.csv", index=False, encoding="utf-8-sig")
    print(f"시작 시각이 없어 분포에서 뺀 제품 {int(feats['start_time'].isna().sum()):,}대")
    print(f"집계표 3개 저장 (곡선 {len(curve)}행, 경로 {len(visit)}행, 시각 {len(hist)}구간)")

    parts = scored_parts(feats, y, s_lgb)
    parts.to_parquet(args.artifact_dir / "scored_parts.parquet", index=False)
    print(f"로컬 전용 {args.artifact_dir}/scored_parts.parquet {len(parts):,}행 (커밋 안 함)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-19 작성. 제품 단위 상위 기여 공정(SHAP)은 넣지 않음.
#     시험 구간 23만 행 전체에 SHAP 을 돌려야 해서 W5 범위를 넘음.
#     화면 드릴다운은 방문 경로·공정 흐름 값까지만 보여 줌
####################################################################
