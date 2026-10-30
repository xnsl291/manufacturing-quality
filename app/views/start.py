####################################################################
# start.py — 0. 시작: 링크를 처음 연 사람에게 대표 메시지를 먼저 보여 줌
#
# 가장 많이 볼 사람은 30초~3분만 보는 면접관·검토자라, 첫 화면에서
# "검사 물량을 얼마로 정할까"라는 판단과 1% 지점의 숫자가 바로 읽혀야 함
# (화면 시안 A1, 2026-10-30 채택). 숫자는 1번 페이지와 같은 집계표에서 읽음.
####################################################################

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import at_rate, header, require  # noqa: E402

# 제목에 숫자가 들어가므로 표를 먼저 읽되, 표가 없어도 제목·안내 문구는 먼저 보여 줌 (명세 0절 4번)
try:
    row, test = at_rate(0.01)
except (FileNotFoundError, IndexError):
    row = None
st.caption("검사 물량을 얼마로 정할까")
header(f"전체의 1%만 더 보면 불량의 {row['포착률']:.1%}를 잡습니다" if row is not None else "품질 Risk Monitoring")
require("T_포착률곡선", "T_구간별불량률")
if row is None:
    st.error("T_포착률곡선.csv 에 검사 물량 1% 행이 없음. 먼저 src/build_artifacts.py 를 돌릴 것")
    st.stop()
recall, lift = f"{row['포착률']:.1%}", f"{row['Lift']:.1f}배"

hero, card = st.columns([2, 1], gap="large")
hero.markdown(
    "무작위로 1%를 뽑아 검사하면 불량도 1%만 잡힙니다. 같은 물량을 위험 순위대로 보면 "
    f"**무작위의 {lift}**를 잡습니다. 물량을 늘릴수록 더 많이 잡지만 효율은 떨어집니다 — "
    "그 맞교환을 한 화면에서 정합니다."
)
hero.page_link("views/overview.py", label="1. 품질 개요 열기", icon=":material/arrow_forward:")
hero.page_link("views/diagnostics.py", label="3. 근거부터 보기 — 모델·공정 진단", icon=":material/fact_check:")

with card.container(border=True):
    st.metric("검사 물량 1%일 때 잡는 불량", recall, delta=f"무작위의 {lift}", delta_color="off")
    # 좁은 카드에 metric 3칸을 두면 '2,368대' 가 '2,36…' 으로 잘려 한 줄 글로 둠
    st.markdown(
        f"검사 대상 **{int(row['검사_대수']):,}대** · 그중 실제 불량 **{int(row['포착_불량'])}건** · "
        f"정밀도 **{row['정밀도']:.1%}**"
    )

s = st.columns(3)
s[0].metric("분석 대수 (시험 구간)", f"{int(test['제품수']):,}대")
s[1].metric("실제 불량률", f"{test['불량률']:.3%}", help=f"불량 {int(test['불량수']):,}건")
s[2].metric("시험 구간 기준", "가장 늦은 20%", help="과거로 학습해 이후 생산품을 판정하는 조건")

PAGES = [
    ("1. 품질 개요", "검사 물량을 슬라이더로 정하면 몇 대를 검사하고 불량의 몇 %를 잡는지 바로 보여 줍니다."),
    ("2. 위험 모니터 · 로컬 전용", "정한 물량만큼 실제로 검사할 제품 목록과 제품 한 대의 정보입니다. 공개 저장소에서는 열리지 않습니다."),
    ("3. 모델·공정 진단", "평가 조건, 임계값 운영, 예측 시점, 모델이 반응한 공정, 놓친 불량까지의 근거입니다."),
]
for col, (name, text) in zip(st.columns(3), PAGES):
    with col.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(text)

st.caption(
    "데이터: Bosch Production Line Performance (Kaggle, 2016) · 시험 구간은 시간상 가장 늦은 20% · "
    "제품 단위 화면(2번)은 재배포 금지라 로컬에서만 동작함 · 코드 github.com/xnsl291/manufacturing-quality"
)
