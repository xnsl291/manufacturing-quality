####################################################################
# common.py — 세 페이지가 같이 쓰는 표 읽기·안내 문구
#
# 화면은 원본을 읽지 않는다. reports/ 집계표와 runs/ 요약만 읽고,
# Risk Monitor 만 로컬 전용 artifacts/scored_parts.parquet 를 읽는다.
# 표 읽기는 src/figures.load 를 그대로 쓴다 — 그림 스크립트와 화면이
# 같은 표를 같은 방식으로 읽어야 숫자가 어긋나지 않는다.
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


@st.cache_data
def table(name: str, **kw) -> pd.DataFrame:
    return figures.load(name, **kw)


@st.cache_data
def inflation() -> pd.DataFrame:
    return figures.inflation_table()


@st.cache_data
def run_json(name: str) -> dict:
    return json.loads((figures.RUNS / name).read_text(encoding="utf-8"))


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
        st.stop()


def show(fig: plt.Figure, where=st) -> None:
    """그림을 띄우고 닫음. st.pyplot 은 받은 그림을 닫지 않아 다시 그릴 때마다 쌓임"""
    where.pyplot(fig)
    plt.close(fig)


def header(title: str) -> None:
    st.title(title)
    st.caption(BANNER)
