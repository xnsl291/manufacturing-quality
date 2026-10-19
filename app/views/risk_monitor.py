####################################################################
# risk_monitor.py — 2. 위험 모니터: 어떤 제품부터 확인할지 목록
#
# 로컬 전용
#   제품 한 대마다의 위험도·공정 흐름 값은 원본에서 나온 데이터라 대회
#   규정의 재배포 금지에 걸린다. 그래서 이 페이지는 로컬에서 만든
#   artifacts/scored_parts.parquet 가 있을 때만 뜨고, 캡처를 공개하지
#   않는다 (2026-10-19 결정). 파일이 없으면 안내만 띄운다.
#
# 임계값  검사 물량을 고르면 그 분위수를 임계값으로 쓴다. 확률값을 고정
#   하지 않는 이유는 3번 페이지 임계값 안정성 그림 — 기간이 바뀌면
#   검사 물량이 2배 가까이 넘친다.
####################################################################

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import header  # noqa: E402
from src.metrics import threshold_at_alert_rate  # noqa: E402

header("2. 위험 모니터 (로컬 전용)")
st.warning("제품 단위 화면임. 대회 규정상 재배포 금지 데이터에서 나온 값이라 **캡처·공유하지 말 것.**")

ARTIFACT = common.ARTIFACT  # 테스트가 바꿔 끼울 수 있게 모듈 속성으로 읽음
if not ARTIFACT.exists():
    st.info(
        "artifacts/scored_parts.parquet 가 없음. 원본 데이터가 있는 PC 에서 "
        "`python src/build_artifacts.py` 를 먼저 돌릴 것. 1·3번 페이지는 이 파일 없이 동작함."
    )
    st.stop()


@st.cache_data
def parts(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["방문_목록"] = df["방문_공정"].str.split("|")
    return df


df = parts(ARTIFACT)
all_stations = sorted({s for p in df["방문_목록"] for s in p if s})
with st.sidebar:
    rate = st.slider("검사 물량 (%)", 0.1, 10.0, 1.0, 0.1) / 100
    outcome = st.radio("실제 결과", ["전체", "불량만", "양품만"], horizontal=True)
    must = st.selectbox("이 공정을 거친 제품만", ["(제한 없음)"] + all_stations)

# 1번 페이지 집계표와 같은 함수로 임계값을 구함 — 따로 계산하면 숫자가 어긋날 수 있음
thr = threshold_at_alert_rate(df["위험도"].to_numpy(), rate)
alert = df[df["위험도"] >= thr]
n_defect = int(df["실제_불량"].sum())
c = st.columns(4)
c[0].metric("검사 대상", f"{len(alert):,}대", help=f"임계값 {thr:.4f} (검사 물량의 분위수)")
c[1].metric("그중 실제 불량", f"{int(alert['실제_불량'].sum()):,}건", help=f"시험 구간 불량 {n_defect:,}건 중")
c[2].metric("불량 포착률", f"{alert['실제_불량'].sum() / max(n_defect, 1):.1%}")
c[3].metric("정밀도", f"{alert['실제_불량'].mean():.1%}" if len(alert) else "-")

view = alert
if outcome != "전체":
    view = view[view["실제_불량"] == (1 if outcome == "불량만" else 0)]
if must != "(제한 없음)":
    view = view[view["방문_목록"].apply(lambda s: must in s)]
view = view.sort_values("위험도", ascending=False)

cols = {
    "Id": "제품 Id",
    "위험도": "위험도",
    "실제_불량": "실제 불량",
    "duration": "공정 소요시간",
    "visited_station_count": "방문 공정 수",
    "path_id": "경로 묶음",
}
st.dataframe(view[list(cols)].rename(columns=cols).head(500), hide_index=True, width="stretch")
st.caption(f"조건에 맞는 검사 대상 {len(view):,}대 중 위험도 상위 500대까지 표시. 경로 묶음 0 은 희귀 경로")

if len(view):
    pick = st.selectbox("자세히 볼 제품 Id", view["Id"].head(500))
    r = view.loc[view["Id"] == pick].iloc[0]
    st.subheader(f"제품 {pick}")
    d = st.columns(4)
    d[0].metric("위험도", f"{r['위험도']:.4f}")
    d[1].metric("실제 결과", "불량" if r["실제_불량"] else "양품")
    d[2].metric("공정 소요시간", f"{r['duration']:.2f}")
    d[3].metric("경로 묶음", "기타(희귀)" if r["path_id"] == 0 else f"경로 {int(r['path_id']):02d}")
    st.write("방문 공정: " + " → ".join(r["방문_목록"]))
    st.caption(
        "위험도는 모델 점수의 순위일 뿐 불량 원인이 아님. 어느 공정에 반응했는지는 "
        "3번 페이지의 공정 기여도(집계)를 볼 것. 제품 단위 기여도는 계산하지 않았음"
    )
