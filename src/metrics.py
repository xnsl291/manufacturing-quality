####################################################################
# metrics.py — 이 프로젝트의 결론을 만드는 평가부
#
# 왜 이 파일이 가장 중요한가
#   이 프로젝트의 결론은 "모델 정확도가 몇 %인가"가 아니라
#   **"전체 생산품 중 몇 %를 추가로 확인하면 실제 불량의 몇 %를 잡는가,
#   그리고 그게 무작위 표본검사보다 몇 배 나은가"** 임.
#   그 표와 숫자를 만드는 곳이 여기다.
#
# Lift 가 왜 핵심인가
#   무작위 표본검사는 정의상 **검사율 = 포착률**이다. 전체의 1%를
#   무작위로 뽑으면 불량의 1%를 잡는다. 따라서
#       Lift = 불량 포착률 ÷ 검사율
#   이고, 검사율 1%에서 포착률 30%면 무작위 대비 30배다.
#   품질 담당자는 PR-AUC를 모르지만 "같은 인원으로 30배 잡는다"는
#   바로 알아듣는다.
#
# 임계값을 확률 0.5로 두지 않는 이유
#   불량률이 0.58%인 환경에서 0.5는 아무것도 걸리지 않는다. 대신
#   "전체의 몇 %를 볼 수 있나"를 먼저 정하고 그 분위수를 임계값으로
#   쓴다. 현장이 움직이는 방식이 그쪽이다.
####################################################################

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef


####################################################################
# threshold_at_alert_rate — "전체의 a 를 본다"는 조건의 임계값을 구함
#
# 역할  검사 물량을 먼저 정하고 그에 맞는 확률 임계값을 돌려준다.
# 방식  점수의 (1-a) 분위수를 쓴다. 동점이 많으면 실제 알람 수가
#       목표를 넘을 수 있으므로, 부르는 쪽이 실제 알람 수를 함께
#       확인하도록 돌려준다.
####################################################################
def threshold_at_alert_rate(score: np.ndarray, alert_rate: float) -> float:
    if not 0 < alert_rate <= 1:
        raise ValueError(f"alert_rate 는 0 초과 1 이하여야 함: {alert_rate}")
    return float(np.quantile(score, 1.0 - alert_rate))


####################################################################
# threshold_table — 검사 물량별 포착률·정밀도·Lift 표를 만듦
#
# 역할  이 프로젝트의 대표 산출물. 맨 아래에 전수 검사 행을 넣어
#       "전수 대비 몇 %의 노력으로 몇 %를 잡는가"가 한눈에 보이게 한다.
# 방식  목표 검사율마다 임계값을 구하고, 그 임계값 이상을 알람으로 본다.
#       **실제 알람율**을 따로 계산해 목표와 어긋나는지 드러낸다 —
#       동점이 많으면 목표 1%인데 실제 2%가 되는 일이 실제로 생긴다.
#       Lift 는 실제 알람율로 나눈다. 목표로 나누면 어긋난 만큼 거짓이 된다.
####################################################################
def threshold_table(
    y_true: np.ndarray,
    score: np.ndarray,
    alert_rates: tuple[float, ...] = (0.005, 0.01, 0.03, 0.05, 0.10),
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype="float64")
    n = len(y_true)
    n_defect = int(y_true.sum())
    base_rate = n_defect / n if n else float("nan")

    rows = []
    for target in alert_rates:
        thr = threshold_at_alert_rate(score, target)
        alert = score >= thr
        n_alert = int(alert.sum())
        caught = int(y_true[alert].sum())
        actual_rate = n_alert / n
        recall = caught / n_defect if n_defect else float("nan")
        rows.append(
            {
                "목표_검사율": target,
                "임계값": thr,
                "실제_검사율": actual_rate,
                "검사_대수": n_alert,
                "포착_불량": caught,
                "포착률": recall,
                "정밀도": caught / n_alert if n_alert else float("nan"),
                "Lift": recall / actual_rate if actual_rate else float("nan"),
                "MCC": _safe_mcc(y_true, alert),
            }
        )

    # 전수 검사 — 비교 기준선
    rows.append(
        {
            "목표_검사율": 1.0,
            "임계값": float(score.min()),
            "실제_검사율": 1.0,
            "검사_대수": n,
            "포착_불량": n_defect,
            "포착률": 1.0,
            "정밀도": base_rate,
            "Lift": 1.0,
            "MCC": float("nan"),
        }
    )
    return pd.DataFrame(rows)


####################################################################
# _safe_mcc — 한쪽 클래스만 예측된 경우에도 터지지 않게 함
#
# 역할  MCC 는 예측이 전부 같은 값이면 분모가 0이 되어 경고와 함께
#       0을 돌려준다. 그 0을 "성능이 0"으로 오해하지 않도록 nan 으로
#       명시해 구분한다.
####################################################################
def _safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(matthews_corrcoef(y_true, y_pred))


####################################################################
# summary — 모델 하나의 지표를 한 줄로 요약
#
# 역할  모델 비교표의 한 행을 만든다. 정확도를 **일부러 같이 넣는다** —
#       불량률 0.58% 환경에서 정확도 99.4%가 왜 무의미한지 보이는 것이
#       Dummy 모델을 두는 목적이기 때문이다.
####################################################################
def summary(y_true: np.ndarray, score: np.ndarray, at_alert_rate: float = 0.01) -> dict:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype="float64")
    thr = threshold_at_alert_rate(score, at_alert_rate)
    alert = score >= thr
    n_defect = int(y_true.sum())
    caught = int(y_true[alert].sum())
    actual_rate = float(alert.mean())
    recall = caught / n_defect if n_defect else float("nan")

    return {
        "PR_AUC": float(average_precision_score(y_true, score)) if n_defect else float("nan"),
        f"포착률@{at_alert_rate:.0%}": recall,
        f"정밀도@{at_alert_rate:.0%}": caught / int(alert.sum()) if alert.any() else float("nan"),
        f"Lift@{at_alert_rate:.0%}": recall / actual_rate if actual_rate else float("nan"),
        f"실제검사율@{at_alert_rate:.0%}": actual_rate,
        f"MCC@{at_alert_rate:.0%}": _safe_mcc(y_true, alert),
        "정확도(참고용)": float((alert == y_true).mean()),
        "불량률": n_defect / len(y_true) if len(y_true) else float("nan"),
    }


####################################################################
# threshold_stability — 검증에서 정한 임계값을 시험 기간에 쓰면 어떻게 되나
#
# 왜 이것이 필요한가 — Week 1에서 알게 된 것
#   실제 데이터의 구간별 불량률이 학습 0.730% / 검증 0.308% / 시험
#   0.407% 로 최대 2.4배 차이였다. 공정 상태가 기간마다 달라지면
#   확률 분포도 밀린다. 그러면 검증에서 "확률 0.0143 이상을 알람"으로
#   정해 둔 것이 시험 기간에는 목표 1%가 아니라 2%를 알람하게 된다.
#   현장에서는 검사 인력이 초과된다. 성능 문제가 아니라 운영 문제다.
#
# 두 방식을 비교한다
#   고정 임계값   검증에서 구한 확률값을 그대로 쓴다 → 검사 물량이 흔들림
#   이동 분위수   시험 구간에서 다시 상위 a 를 고른다 → 검사 물량이 고정됨
####################################################################
def threshold_stability(
    y_valid: np.ndarray,
    score_valid: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
    alert_rates: tuple[float, ...] = (0.01, 0.03, 0.05),
) -> pd.DataFrame:
    rows = []
    for a in alert_rates:
        thr = threshold_at_alert_rate(score_valid, a)

        fixed = np.asarray(score_test) >= thr
        fixed_rate = float(fixed.mean())
        fixed_recall = (
            float(np.asarray(y_test)[fixed].sum() / np.asarray(y_test).sum())
            if np.asarray(y_test).sum()
            else float("nan")
        )

        thr_moving = threshold_at_alert_rate(score_test, a)
        moving = np.asarray(score_test) >= thr_moving
        moving_rate = float(moving.mean())
        moving_recall = (
            float(np.asarray(y_test)[moving].sum() / np.asarray(y_test).sum())
            if np.asarray(y_test).sum()
            else float("nan")
        )

        rows.append(
            {
                "목표_검사율": a,
                "검증에서_정한_임계값": thr,
                "고정_실제검사율": fixed_rate,
                "고정_검사율_오차배수": fixed_rate / a if a else float("nan"),
                "고정_포착률": fixed_recall,
                "이동분위수_실제검사율": moving_rate,
                "이동분위수_포착률": moving_recall,
            }
        )
    return pd.DataFrame(rows)


####################################################################
# cost_curve — 비용 비율만으로 최적 검사 물량을 봄
#
# 왜 금액을 쓰지 않는가
#   실제 검사비·유출비를 모른다. 그래서 **비율 하나**로 일반화한다.
#       r = (불량 1건 유출 비용) ÷ (1대 더 검사하는 비용)
#       총비용(a) ∝ a + r · p · (1 − R(a))
#   앞이 검사 부담, 뒤가 유출 손실이다. r 을 바꾸며 최적 a 를 찾으면
#   "유출 비용이 검사 비용의 몇 배를 넘으면 검사를 늘리는 게 이득인가"
#   라는 형태의 결론이 나온다. 모른다는 한계를 인정하면서도 의사결정에
#   쓸 수 있는 형태다.
####################################################################
def cost_curve(
    table: pd.DataFrame,
    ratios: tuple[float, ...] = (10, 50, 100, 500, 1000),
    defect_rate: float | None = None,
) -> pd.DataFrame:
    p = defect_rate if defect_rate is not None else float(
        table.loc[table["목표_검사율"] == 1.0, "정밀도"].iloc[0]
    )
    rows = []
    for r in ratios:
        best = None
        for _, t in table.iterrows():
            a = float(t["실제_검사율"])
            recall = float(t["포착률"])
            cost = a + r * p * (1.0 - recall)
            if best is None or cost < best[1]:
                best = (t["목표_검사율"], cost, a, recall)
        rows.append(
            {
                "비용비_r": r,
                "최적_검사율": best[0],
                "그때_포착률": best[3],
                "상대총비용": best[1],
            }
        )
    return pd.DataFrame(rows)


####################################################################
# 작업 메모
#   - 2026-10-02 작성. Lift 를 **실제 검사율**로 나누는 것이 핵심임.
#     목표 검사율로 나누면 동점 때문에 어긋난 만큼 Lift 가 거짓이 됨
####################################################################
