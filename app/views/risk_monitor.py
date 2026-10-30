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
from common import LOCAL_NOTICE, TICKS, header, tick_label  # noqa: E402
from src.metrics import threshold_at_alert_rate  # noqa: E402

header("2. 위험 모니터 (로컬 전용)")
st.warning("제품 단위 화면임. 대회 규정상 재배포 금지 데이터에서 나온 값이라 **캡처·공유하지 말 것.**")

ARTIFACT = common.ARTIFACT  # 테스트가 바꿔 끼울 수 있게 모듈 속성으로 읽음
if not ARTIFACT.exists():
    st.info(
        "artifacts/scored_parts.parquet 가 없음. 원본 데이터가 있는 PC 에서 "
        "`python src/build_artifacts.py` 를 먼저 돌릴 것. 1·3번 페이지는 이 파일 없이 동작함."
    )
    st.session_state.pop("handoff_rate", None)
    st.stop()

# CSV 내려받기 차단과 이 PC 전용 접속은 app/.streamlit/config.toml 이 읽혀야 걸림. 설정이 빠진 채
# 떴으면(다른 설정 파일이 덮어씀 등) 제품 값을 그리지 않음. disableDataExport 는 Streamlit 설명대로
# 보안 장치가 아니라 실수 방지용이고, 진짜 경계는 '이 PC 밖으로 내보내지 않음'임
if st.get_option("server.address") not in ("localhost", "127.0.0.1") or not st.get_option("client.disableDataExport"):
    st.error(
        "app/.streamlit/config.toml 설정(이 PC 전용 접속·내려받기 차단)이 적용되지 않아 목록을 열지 않음. "
        "저장소의 설정 파일을 지우거나 덮어쓰지 않았는지 확인할 것"
    )
    st.stop()


# 처음 들어오면 로컬 전용이라는 것을 한 번 확인받음 (화면 시안 B2). 확인 전에는 목록·KPI 를
# 그리지 않음 — 창 뒤에 제품 값이 먼저 보이면 확인을 받는 의미가 없음. 세션 동안 한 번만 물음
@st.dialog("로컬 전용 화면입니다", dismissible=False)
def confirm_local() -> None:
    st.write(LOCAL_NOTICE)
    a, b = st.columns(2)
    if a.button("1번으로 돌아가기", width="stretch"):
        st.session_state.pop("handoff_rate", None)
        st.switch_page("views/overview.py")
    if b.button("이해했습니다, 열기", type="primary", width="stretch"):
        st.session_state["local_ok"] = True
        st.rerun()


if not st.session_state.get("local_ok"):
    confirm_local()
    st.stop()


@st.cache_data
def parts(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["방문_목록"] = df["방문_공정"].str.split("|")
    return df


try:
    df = parts(ARTIFACT)
except Exception as e:  # 깨진 파일·만드는 중인 파일·열 이름이 다른 옛 파일 — 무엇이든 traceback 대신 안내
    st.error(
        "artifacts/scored_parts.parquet 를 읽을 수 없음 — 파일이 깨졌거나 아직 만드는 중일 수 있음. "
        f"목록과 KPI 는 그리지 않음. ({type(e).__name__})"
    )
    with st.container(border=True):
        st.markdown(
            "**무엇을 하면 되나**\n"
            "- 원본 데이터가 있는 PC 에서 `python src/build_artifacts.py` 를 다시 돌리면 이 파일이 새로 만들어짐\n"
            "- 파일을 만드는 중이라면 끝난 뒤 새로고침하면 됨\n"
            "- 1번 품질 개요와 3번 모델·공정 진단은 이 파일 없이도 그대로 동작함"
        )
        st.page_link("views/overview.py", label="1. 품질 개요로 가기", icon=":material/arrow_forward:")
    st.stop()

all_stations = sorted({s for p in df["방문_목록"] for s in p if s})
# 1번에서 넘겨받은 물량이 있으면 그 값으로 시작함 (화면 시안 B1). 위젯을 만들기 전에 넣어야 함.
# 눈금에 없는 값이면 select_slider 가 조용히 첫 칸(0.1%)으로 돌아가므로 가장 가까운 눈금으로 맞춤.
# 값이 남아 있을 때 덮어쓰면 사용자가 방금 옮긴 슬라이더가 되돌아가므로 비었을 때만 1% 로 채움
if "handoff_rate" in st.session_state:
    v = st.session_state.pop("handoff_rate")
    st.session_state["monitor_rate"] = min(TICKS, key=lambda t: abs(t - v))
elif "monitor_rate" not in st.session_state:
    st.session_state["monitor_rate"] = 0.01
with st.sidebar:
    # persist_state: 다른 페이지에 다녀와도 물량이 1% 로 돌아가지 않게 함 (2026-10-30 브라우저 확인)
    rate = st.select_slider(
        "검사 물량", options=TICKS, format_func=tick_label, key="monitor_rate", persist_state="session"
    )
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
n_view = len(view)
view = view.sort_values("위험도", ascending=False).head(500)

cols = {
    "Id": "제품 Id",
    "위험도": "위험도",
    "실제_불량": "실제 불량",
    "duration": "공정 소요시간",
    "visited_station_count": "방문 공정 수",
    "path_id": "경로 묶음",
}
listing, detail = st.columns([2, 1], gap="medium")
with listing:
    # 조건이 바뀌면 같은 행 번호가 다른 제품을 가리키므로, 조건마다 표 key 를 달리해 선택을 풀어 줌
    event = st.dataframe(
        view[list(cols)].rename(columns=cols),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"parts_{rate}_{outcome}_{must}",
        column_config={
            "위험도": st.column_config.NumberColumn(format="%.4f"),
            "공정 소요시간": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        f"조건에 맞는 검사 대상 {n_view:,}대 중 위험도 상위 500대까지 표시. 경로 묶음 0 은 희귀 경로. "
        "행을 누르면 오른쪽에 상세가 뜸"
    )

with detail:
    rows = event.selection.rows
    if not rows:
        st.info("목록에서 행을 누르면 여기에 제품 상세가 뜸" if n_view else "조건에 맞는 제품이 없음")
    else:
        r = view.iloc[rows[0]]
        with st.container(border=True):
            st.subheader(f"제품 {r['Id']}")
            d = st.columns(2)
            d[0].metric("위험도", f"{r['위험도']:.4f}")
            d[1].metric("실제 결과", "불량" if r["실제_불량"] else "양품")
            d = st.columns(2)
            d[0].metric("공정 소요시간", f"{r['duration']:.2f}")
            d[1].metric("경로 묶음", "기타(희귀)" if r["path_id"] == 0 else f"경로 {int(r['path_id']):02d}")
            # 화살표는 위험도 카드 옆에서 '이 흐름 때문에 위험' 으로 읽힐 수 있어 가운데점으로 나열함 (명세 0절 5번)
            st.write("방문 순서: " + " · ".join(r["방문_목록"]))
            st.caption(
                "위험도는 모델 점수의 순위일 뿐 불량 원인이 아님. 어느 공정에 반응했는지는 "
                "3번 페이지의 공정 기여도(집계)를 볼 것. 제품 단위 기여도는 계산하지 않았음"
            )
            st.page_link("views/diagnostics.py", label="3. 모델·공정 진단에서 공정 기여도 보기", icon=":material/arrow_forward:")
