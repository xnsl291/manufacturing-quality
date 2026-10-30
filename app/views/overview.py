####################################################################
# overview.py — 1. 품질 개요: "전체의 몇 %를 보면 불량의 몇 %를 잡나"
#
# 핵심 상호작용은 검사 물량 하나. 고르면 KPI 와 대표 그림의 점이 같이
# 움직인다. 제품 단위 값은 쓰지 않는다 (집계표만).
####################################################################

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ARTIFACT, LOCAL_NOTICE, TICKS, figures, header, require, show, table, tick_label  # noqa: E402

header("1. 품질 개요")
require("T_포착률곡선", "T_구간별불량률", "T_시작시각_분포")

curve = table("T_포착률곡선")
split = table("T_구간별불량률", index_col=0)
main = curve[curve["모델"] == "LightGBM"].reset_index(drop=True)

# 눈금은 운영에서 실제로 고를 만한 7칸만 보여 줌 (common.TICKS). 곡선 표의 65칸을 그대로 쓰면
#   ① 0.500% 와 0.504% 가 둘 다 "0.50%" 로 보여 같은 라벨이 두 칸 연속 나왔고
#   ② 0.00944 처럼 뜻을 읽을 수 없는 값에 손잡이가 멈췄음
# 후보는 곡선 표에 그대로 들어 있는 값만 씀 (없는 값을 고르면 KPI 조회가 비어 버림).
# 눈금이 하나라도 빠지면 2번 슬라이더와 칸이 달라져 넘긴 물량이 어긋나므로 멈춤
options = [r for r in main["목표_검사율"] if round(r, 5) in TICKS]
if len(options) != len(TICKS):
    st.error("T_포착률곡선.csv 에 검사 물량 눈금 값이 빠져 있음. 먼저 src/build_artifacts.py 를 돌릴 것")
    st.stop()

rate = st.select_slider(
    "검사 물량 — 전체 생산품 중 추가로 확인할 비율",
    options=options,
    value=min(options, key=lambda r: abs(r - 0.01)),
    format_func=tick_label,
)
row = main.loc[main["목표_검사율"] == rate].iloc[0]
test = split.loc["test"]

c = st.columns(4)
c[0].metric("분석 대수 (시험 구간)", f"{int(test['제품수']):,}대")
c[1].metric("실제 불량률", f"{test['불량률']:.3%}", help=f"불량 {int(test['불량수']):,}건")
c[2].metric("선택한 검사 물량", f"{int(row['검사_대수']):,}대", help=f"실제 검사율 {row['실제_검사율']:.2%}")
c[3].metric(
    "그때 잡는 불량",
    f"{row['포착률']:.1%}",
    delta=f"무작위의 {row['Lift']:.1f}배",
    delta_color="off",
    help=f"{int(row['포착_불량'])}건 / 정밀도 {row['정밀도']:.1%}",
)


# 이 창에서 로컬 전용 확인까지 같이 받음. 따로 받으면 1번에서 넘어갈 때 확인 창이 연달아 두 번 뜸
# (2026-10-30 검토). 사이드바로 2번에 바로 들어오면 2번 페이지가 같은 문구로 한 번 물음
@st.dialog("이 검사 물량으로 위험 모니터를 열까요?")
def confirm_handoff() -> None:
    st.write(f"검사 물량 {tick_label(rate)}({int(row['검사_대수']):,}대) 기준으로 2번 위험 모니터를 엽니다.")
    st.info(LOCAL_NOTICE)
    a, b = st.columns(2)
    if a.button("취소", width="stretch"):
        st.rerun()
    if b.button("이해했습니다, 열기", type="primary", width="stretch"):
        st.session_state["handoff_rate"] = round(rate, 5)
        st.session_state["local_ok"] = True
        st.switch_page("views/risk_monitor.py")


# 2번 화면용 로컬 파일이 없는 PC(공개 저장소를 받은 경우)에서는 넘겨도 빈 화면뿐이라 버튼 대신 안내만 둠
if ARTIFACT.exists():
    if st.button(f"이 물량({tick_label(rate)})으로 검사 목록 보기 — 2. 위험 모니터", icon=":material/arrow_forward:"):
        confirm_handoff()
else:
    st.caption("이 물량의 제품 목록(2. 위험 모니터)은 원본 데이터가 있는 PC 에서만 열림 — 재배포 금지 데이터")

show(figures.capture_curve(curve, highlight=rate))
st.markdown(
    "- 점선이 무작위 표본검사임. 무작위로 1%를 보면 불량도 1%만 잡음\n"
    "- 시험 구간은 시간상 가장 늦은 20%임. 과거로 학습해 이후 생산품을 판정하는 조건이라, "
    "무작위로 나눈 공개 결과보다 숫자가 낮음 (3번 페이지의 부풀림 그림 참고)"
)

st.subheader("생산 기간과 불량률")
show(figures.f4_start_time(table("T_시작시각_분포")))
st.caption(
    "구간별 불량률 — 학습 {:.3%} / 검증 {:.3%} / 시험 {:.3%}. 기간마다 공정 상태가 달라 "
    "시간 순서로 나눠 평가함".format(*split["불량률"])
)
