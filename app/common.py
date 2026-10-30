####################################################################
# common.py — 네 페이지가 같이 쓰는 표 읽기·페이지 머리·안내 창
#
# 화면은 원본을 읽지 않는다. reports/ 집계표와 runs/ 요약만 읽고,
# Risk Monitor 만 로컬 전용 artifacts/scored_parts.parquet 를 읽는다.
# 표 읽기는 src/figures.load 를 그대로 쓴다 — 그림 스크립트와 화면이
# 같은 표를 같은 방식으로 읽어야 숫자가 어긋나지 않는다.
#
# 대화 상자(st.dialog)는 함수 단위로만 띄울 수 있어 창마다 함수를 하나씩 둔다.
####################################################################

from __future__ import annotations

import sys
from pathlib import Path

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import figures  # noqa: E402

ARTIFACT = ROOT / "artifacts" / "scored_parts.parquet"

BANNER = (
    "이 화면은 공개 데이터(Bosch Production Line Performance, Kaggle 2016)로 만든 **오프라인 분석**임. "
    "실제 출하 판정이나 불량 원인 규명이 아님. 위험도는 보정하지 않은 순위용 점수임."
)

# 검사 물량 눈금. 1번과 2번이 같은 눈금을 써야 1번에서 정한 물량을 2번으로 그대로 넘기고,
# 두 페이지 숫자를 눈금마다 대조할 수 있음 (2026-10-30 결정).
# 곡선 표에 반올림 없이 들어 있는 값이 이 일곱뿐이라 이 일곱으로 정함 (2026-10-29)
TICKS = (0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.20)

# 집계표를 처음부터 다시 만드는 순서 (README 1-2절과 같음). 뒤 단계가 앞 단계 산출물을 읽음
BUILD_STEPS = """\
python src/ingest.py                # 원본 변환 (원본 CSV 필요)
python src/profile_data.py          # 데이터 구조 표
python src/features.py              # T_구간별불량률
python src/train.py                 # T_모델비교 · T_누설실험 · T_임계값안정성 · T_비용비곡선
python src/early_prediction.py      # T_조기예측곡선
python src/error_analysis.py        # T_오탐미탐_집계 · T_오탐미탐_시기비중
python -W ignore src/explain.py     # T_ablation · T_공정기여도 · explain.json
python tools/diagnose_week2.py      # T_Week2진단
python tools/where_is_signal.py     # where_is_signal.json
python src/build_artifacts.py       # T_포착률곡선 · T_시작시각_분포 · 2번 페이지용 로컬 파일"""


# 2번 페이지를 열기 전에 한 번 확인받는 문구. 1번의 넘기기 창과 2번의 첫 진입 창이 같은 문구를 씀
LOCAL_NOTICE = (
    "2번 페이지는 제품 한 대마다의 위험도를 보여 줍니다. 이 값은 대회 규정상 재배포가 금지된 데이터"
    "(Bosch Production Line Performance, Kaggle 2016)에서 나온 것이라 공개 저장소·포트폴리오에 올릴 수 없습니다.\n\n"
    "외부에 보여 줄 자료에는 1·3번 페이지의 집계만 쓰세요. 위험도는 보정하지 않은 순위용 점수이고, 실제 출하 판정이 아닙니다."
)


def tick_label(r: float) -> str:
    """0.001 → '0.1%', 0.01 → '1%'. 끝자리 0 을 떼어 눈금을 읽기 쉽게 함"""
    return f"{r * 100:.2f}".rstrip("0").rstrip(".") + "%"


@st.cache_data
def table(name: str, **kw) -> pd.DataFrame:
    return figures.load(name, **kw)


@st.cache_data
def inflation() -> pd.DataFrame:
    return figures.inflation_table()


@st.cache_data
def run_json(name: str) -> dict:
    return json.loads((figures.RUNS / name).read_text(encoding="utf-8"))


def at_rate(rate: float) -> tuple[pd.Series, pd.Series]:
    """검사 물량 하나에서의 대표 모델 곡선 한 줄과 시험 구간 요약 한 줄.
    시작 화면·1번 KPI·용어 설명이 같은 숫자를 쓰게 한 곳에서 읽음"""
    curve = table("T_포착률곡선")
    main = curve[curve["모델"] == "LightGBM"]
    # 가장 가까운 값을 쓰면 표에 1% 가 없어도 '1%' 라고 적은 채 다른 물량의 숫자를 보여 주게 됨.
    # 정확히 같은 행만 쓰고, 없으면 IndexError 로 멈춤
    row = main[main["목표_검사율"].round(5) == rate].iloc[0]
    return row, table("T_구간별불량률", index_col=0).loc["test"]


def require(*names: str) -> None:
    """필요한 집계표(.csv)·실행 요약(.json)이 없으면 무엇을 돌려야 하는지 보여 주고 멈춤"""
    missing = []
    for n in names:
        path = figures.RUNS / n if n.endswith(".json") else figures.REPORTS / f"{n}.csv"
        if not path.exists():
            missing.append((path.name, figures.SOURCE.get(n, "README 1-2절의 실행 순서")))
    if missing:
        for name, src in missing:
            st.error(f"{name} 가 없음. 먼저 {src} 를 돌릴 것")
        if st.button("집계표 만드는 법 보기"):
            how_to_build()
        st.stop()


@st.dialog("집계표가 없을 때 — 무엇을 하면 되나", width="large")
def how_to_build() -> None:
    st.markdown(
        "**공개 저장소를 받았다면** 집계표는 저장소에 들어 있음. 지워졌거나 바뀐 것이므로 "
        "저장소 맨 위 폴더에서 아래 명령으로 되살리면 됨 (원본 데이터 불필요)"
    )
    st.code("git checkout -- reports runs", language="bash")
    st.markdown(
        "**처음부터 다시 만들려면** 원본 데이터(Kaggle 로그인·대회 규정 동의 필요)가 있어야 함. "
        "뒤 단계가 앞 단계 산출물을 읽으므로 이 순서를 지킬 것. 오른쪽 위 아이콘으로 복사할 수 있음"
    )
    st.code(BUILD_STEPS, language="bash")
    st.caption("2번 위험 모니터는 마지막 단계가 만드는 로컬 파일이 있어야 열림. 전체 순서와 소요 시간은 README 1-2절")


@st.dialog("용어 설명 — 이 화면의 숫자를 읽는 법")
def glossary() -> None:
    try:
        row, test = at_rate(0.01)
        ex_recall = f" 1% 물량에서 {row['포착률']:.1%}({int(row['포착_불량'])}건)."
        ex_lift = f" 무작위로 1%를 보면 불량도 1%만 잡히므로 {row['포착률']:.1%} ÷ 1% = {row['Lift']:.1f}배."
        ex_prec = f" 1% 물량({int(row['검사_대수']):,}대)에서 {row['정밀도']:.1%}."
        total = f" {int(test['불량수']):,}건"
    except (FileNotFoundError, IndexError):  # 집계표가 없거나 1% 행이 없어도 용어 뜻은 보여 줌
        ex_recall = ex_lift = ex_prec = total = ""
    st.markdown(
        f"**불량 포착률** — 시험 구간의 전체 불량{total} 중, 고른 검사 물량 안에서 몇 %를 잡았는지.{ex_recall}\n\n"
        f"**무작위 대비 배수 (Lift)** — 같은 물량을 무작위로 뽑았을 때와 비교한 값.{ex_lift}\n\n"
        f"**정밀도** — 검사한 제품 중 실제 불량이 몇 %였는지.{ex_prec} "
        "물량을 늘리면 포착률은 오르고 정밀도는 떨어짐.\n\n"
        "**위험도** — 순위를 매기기 위한 점수. 보정하지 않았으므로 확률(%)로 읽으면 안 됨."
    )
    a, b = st.columns(2)
    if a.button("닫기", width="stretch"):
        st.rerun()
    if b.button("3번 진단 페이지에서 근거 보기", type="primary", width="stretch"):
        st.switch_page("views/diagnostics.py")


def show(fig: plt.Figure, where=st) -> None:
    """그림을 띄우고 닫음. st.pyplot 은 받은 그림을 닫지 않아 다시 그릴 때마다 쌓임"""
    where.pyplot(fig)
    plt.close(fig)


def header(title: str) -> None:
    left, right = st.columns([5, 1], vertical_alignment="bottom")
    left.title(title)
    if right.button("용어 설명", key="glossary", width="stretch"):
        glossary()
    st.caption(BANNER)
