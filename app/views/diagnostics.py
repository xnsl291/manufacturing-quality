####################################################################
# diagnostics.py — 3. 모델·공정 진단: 숫자를 믿어도 되는가
#
# 면접에서 파고드는 질문 순서대로 둔다.
#   모델 비교 → 평가 조건이 숫자를 얼마나 부풀리나 → 임계값 운영 →
#   예측 시점 → 어느 공정에 반응하나 → 놓친 불량은 누구인가
####################################################################

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import figures, header, inflation, require, run_json, show, table  # noqa: E402

header("3. 모델·공정 진단")
require(
    "T_모델비교", "T_누설실험", "T_Week2진단", "T_임계값안정성", "T_비용비곡선",
    "T_조기예측곡선", "T_ablation", "T_공정기여도", "T_오탐미탐_집계", "T_오탐미탐_시기비중",
    "where_is_signal.json", "explain.json",
)
# 주의 문구는 explain.py 가 남긴 실행 요약에서 읽음. src.explain 을 import 하면
# 문구 하나 때문에 lightgbm 까지 불러와 화면이 늦게 뜸

# 본문이 길어(그림 5장·표 5개) 맨 위에 절 바로 가기를 둠 (화면 시안 C2). anchor 는 아래 소제목과 짝
st.markdown(
    "**바로 가기** · "
    "[3-1 모델 비교](#model) · "
    "[3-2 부풀림](#inflation) · "
    "[3-3 임계값 운영](#threshold) · "
    "[3-4 예측 시점](#early) · "
    "[3-5 반응한 공정](#contrib) · "
    "[3-6 놓친 불량](#errors)"
)

st.subheader("모델 3종 비교 — 시험 구간", anchor="model")
# 비율은 %, Lift 는 배로 적음. 0~1 소수를 그대로 두면 읽는 사람이 포착률 0.1151 을
# 머릿속에서 11.5% 로 환산해야 함 (2026-10-29 화면 시안)
comp = table("T_모델비교", index_col=0)
FMT = {
    "학습초": "{:.1f}초", "PR_AUC": "{:.4f}", "포착률@1%": "{:.1%}", "정밀도@1%": "{:.2%}",
    "Lift@1%": "{:.1f}배", "실제검사율@1%": "{:.2%}", "MCC@1%": "{:.4f}",
    "정확도(참고용)": "{:.2%}", "불량률": "{:.2%}",
}
shown = pd.DataFrame(
    {c: comp[c].map(lambda v, f=f: "—" if pd.isna(v) else f.format(v)) for c, f in FMT.items()},
    index=comp.index,
).rename(columns={"학습초": "학습 시간", "PR_AUC": "PR-AUC", "Lift@1%": "무작위 대비@1%"})
# 누설을 포함한 행은 숫자가 제일 좋아서 그냥 두면 1등처럼 보임. 흐리게 + 줄을 그어
# "배제 대상" 이 먼저 읽히게 함
st.dataframe(
    shown.style.apply(
        lambda r: ["color:#8a93a1; text-decoration:line-through"] * len(r)
        if str(r.name).startswith("X_") else [""] * len(r),
        axis=1,
    ),
    width="stretch",
)
st.caption("정확도는 일부러 같이 둠. 전부 양품이라고 답하는 Dummy 도 99.6%라 정확도로는 아무것도 구분되지 않음")

st.subheader("평가 조건을 잘못 잡으면", anchor="inflation")
show(figures.inflation(inflation()))
st.caption(
    "Id 순서 누설은 인접 Id 의 정답을 쓰는 것이라 운영에서 얻을 수 없음 → 배제. "
    "무작위 분할은 기간별 공정 상태 차이를 지워 **PR-AUC 기준 9배** 부풀림 "
    "(같은 조건에서 1% 포착률 기준으로는 2.2배). 몇 배인지는 어느 지표로 재느냐에 따라 다름"
)

st.subheader("임계값 운영", anchor="threshold")
a, b = st.columns(2)
show(figures.threshold_stability(table("T_임계값안정성")), a)
show(figures.cost_ratio(table("T_비용비곡선")), b)

st.subheader("예측 시점을 앞당기면", anchor="early")
show(figures.early_prediction(table("T_조기예측곡선")))

st.subheader("모델이 어느 공정에 반응하나", anchor="contrib")
st.info(run_json("explain.json")["주의"])
contrib = table("T_공정기여도")
how = st.radio(
    "공정 단위로 묶는 방식",
    ["합계", "평균", "상위포함수"],
    horizontal=True,
    help="합계는 측정값이 많은 공정이, 평균은 측정값 1~2개 공정이 유리함. 셋 다 상위인 공정을 봄",
)
# 열 이름이 코드 이름 그대로면 처음 보는 사람이 읽을 수 없음. 뜻은 그대로 두고 이름만 바꿈
CONTRIB_NAMES = {
    "station_key": "공정", "입력수": "입력 수", "상위포함수": "상위 포함 횟수",
    "합계_순위": "합계 순위", "평균_순위": "평균 순위", "상위포함수_순위": "상위포함 순위",
    "최악순위": "세 방식 중 가장 낮은 순위",
}
st.dataframe(
    contrib.sort_values(f"{how}_순위").head(15).rename(columns=CONTRIB_NAMES),
    hide_index=True, width="stretch",
)
show(figures.ablation(table("T_ablation")))
st.caption(
    "SHAP 상위 공정을 빼도 2건, 절벽 구간 공정만 빼면 19건 줄었음. 공정 흐름 값과 절벽 구간 공정을 **함께** 빼야 "
    "111→10건으로 무너짐 — "
    "같은 신호가 두 곳에 겹쳐 있어 SHAP 순위를 '없으면 안 되는 입력'으로 읽으면 틀림"
)

st.subheader("잡은 불량과 놓친 불량 — 검사율 1%", anchor="errors")
prof = table("T_오탐미탐_집계", index_col=0)
mix = table("T_오탐미탐_시기비중", index_col=0)
st.dataframe(prof, width="stretch")
st.dataframe((mix * 100).round(1), width="stretch")
# 비중(잡은 불량이 어느 시기에 몰렸나)과 포착률(그 시기 불량 중 몇 %를 잡았나)은 다름.
# 비중만 보고 "멀어질수록 못 잡는다"고 쓰면 틀리므로 포착률을 같이 계산해 보여 줌
tp = mix.loc["TP_잡음"] * prof.loc["TP_잡음", "제품수"]
fn = mix.loc["FN_놓침"] * prof.loc["FN_놓침", "제품수"]
recall = (tp / (tp + fn) * 100).round(1).rename("시기별 포착률(%)")
st.dataframe(recall.to_frame().T, width="stretch")
# 숫자만 나란히 두면 "시간이 갈수록 못 잡는다" 로 잘못 읽힘(실제로 W3 에 한 번 틀렸던 자리).
# 넷째 시기 바닥과 다섯째 반등이 모양으로도 보이게 막대를 같이 둠
st.bar_chart(recall, height=180)
st.caption(
    "첫 표는 집단별 중앙값, 둘째 표는 시험 구간을 시기 5개로 나눈 비중(%), 셋째 줄은 시기별 포착률임. 잡은 불량의 42%가 첫 시기에 "
    "몰려 있고, 포착률은 넷째 시기까지 줄다가 다섯째 시기에 다시 오름 — 학습 시점에서 멀어질수록 단조롭게 "
    "나빠진다고까지는 말할 수 없음. 집계만 보여 줌 (제품 단위 표 없음)"
)
