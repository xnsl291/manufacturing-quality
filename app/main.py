####################################################################
# main.py — 품질 Risk Monitoring 화면 진입점
#
# 실행  streamlit run app/main.py
# 주소 / 로 들어오면 0. 시작이 뜸. 없는 주소로 들어와도 Streamlit 이 기본
# 페이지(0. 시작)를 대신 띄우므로 따로 안내 화면을 두지 않음 (2026-10-30)
# 페이지 1·3 은 저장소의 집계표만으로 뜬다 (원본 데이터 불필요).
# 페이지 2 는 로컬에서 src/build_artifacts.py 를 돌린 뒤에만 뜬다.
#
# 페이지 폴더를 pages/ 가 아니라 views/ 로 둔 이유
#   pages/ 는 Streamlit 이 자동으로 페이지로 인식한다. 그러면 주소로 바로
#   들어온 경우 이 파일(메뉴 이름·넓은 화면 설정)을 거치지 않고 열려서
#   메뉴가 파일 이름으로 바뀌고 화면 폭이 좁아졌음 (2026-10-19 확인)
####################################################################

import streamlit as st

st.set_page_config(page_title="품질 Risk Monitoring", layout="wide")
nav = st.navigation(
    [
        st.Page("views/start.py", title="0. 시작", default=True),
        st.Page("views/overview.py", title="1. 품질 개요"),
        st.Page("views/risk_monitor.py", title="2. 위험 모니터 (로컬 전용)"),
        st.Page("views/diagnostics.py", title="3. 모델·공정 진단"),
    ]
)
nav.run()
