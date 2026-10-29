####################################################################
# figures.py — 집계표에서 그림을 그림 (Week 5)
#
# 원칙
#   - 원본 데이터를 읽지 않는다. reports/T_*.csv 와 runs/*.json 만 읽는다.
#     그래서 저장소를 받기만 하면 데이터 없이도 그림이 다시 나온다.
#   - 그림 함수는 표를 받아 Figure 를 돌려준다. PNG 저장(이 파일의
#     main)과 화면(app/)이 **같은 함수**를 부른다. 그림과 화면의 숫자가
#     따로 놀지 않게 하려는 것이다.
#   - 제품 단위 행은 어떤 그림에도 넣지 않는다 (대회 규정 재배포 금지).
#
# 그림 목록 (파일명 = 이 그림이 답하는 질문)
#   대표   capture_curve       전체의 몇 %를 보면 불량의 몇 %를 잡나
#   W2     inflation           분할·누설 실수가 성능을 몇 배 부풀리나
#   W3     early_prediction    예측 시점을 앞당기면 얼마나 잃나
#   W3     threshold_stability 확률 임계값을 고정하면 검사 물량이 어떻게 되나
#   W3     cost_ratio          미탐 비용이 몇 배일 때 검사를 얼마나 늘려야 하나
#   W4     ablation            어느 입력 묶음을 빼야 성능이 무너지나
#   W1     F1~F5               데이터 구조 (공정·결측·경로·시간·순서)
####################################################################

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경(테스트·서버)에서도 그려지게 함
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"

LINE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
MAIN = "#1f4e79"
GREY = "#8c8c8c"
# 선택한 지점을 가리키는 색. 빨강은 "나쁜 결과"에만 쓰고,
# "여기를 보라"는 뜻은 색을 나눠 둔다 (2026-10-29 화면 시안)
PICK = "#0e8f8f"


def _set_korean_font() -> None:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# 로그 축 눈금을 10^-1 대신 0.1 로 적음. 한글 글꼴에 지수 표기용 빼기 기호가
# 없어 깨지고, 품질 담당자에게는 0.1 이 더 바로 읽힘
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")


_set_korean_font()


####################################################################
# load — 집계표 하나를 읽음
#
# 역할  모든 그림과 화면이 이 함수로만 표를 읽는다. 표가 없을 때 어떤
#       스크립트를 돌려야 하는지 한 곳에서 알려 주기 위함이다.
####################################################################
SOURCE = {
    "T2_구조": "src/profile_data.py",
    "T3_경로": "src/profile_data.py",
    "T_공정순서추정": "src/profile_data.py",
    "T_구간별불량률": "src/features.py",
    "T_모델비교": "src/train.py",
    "T_누설실험": "src/train.py",
    "T_임계값안정성": "src/train.py",
    "T_비용비곡선": "src/train.py",
    "T_Week2진단": "tools/diagnose_week2.py",
    "T_조기예측곡선": "src/early_prediction.py",
    "T_오탐미탐_집계": "src/error_analysis.py",
    "T_오탐미탐_시기비중": "src/error_analysis.py",
    "T_ablation": "src/explain.py",
    "T_공정기여도": "src/explain.py",
    "T_포착률곡선": "src/build_artifacts.py",
    "T_경로x공정_방문비율": "src/build_artifacts.py",
    "T_시작시각_분포": "src/build_artifacts.py",
    "where_is_signal.json": "tools/where_is_signal.py",
    "explain.json": "src/explain.py",
}


def load(name: str, report_dir: Path = REPORTS, **kw) -> pd.DataFrame:
    path = report_dir / f"{name}.csv"
    if not path.exists():
        src = SOURCE.get(name, "README 1-2절의 실행 순서")
        raise FileNotFoundError(f"{path} 가 없음. 먼저 {src} 를 돌릴 것")
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


####################################################################
# inflation_table — "정직한 조건 대비 몇 배 부풀었나" 표를 조립함
#
# 역할  네 조건이 서로 다른 실행에서 나왔다(train.py 2개, 진단 1개,
#       where_is_signal 1개). README 표와 그림이 같은 숫자를 쓰도록
#       조립을 한 곳에 둔다.
####################################################################
def inflation_table(report_dir: Path = REPORTS, run_dir: Path = RUNS) -> pd.DataFrame:
    leak = load("T_누설실험", report_dir)
    diag = load("T_Week2진단", report_dir, index_col=0)
    sig_path = run_dir / "where_is_signal.json"
    if not sig_path.exists():
        raise FileNotFoundError(f"{sig_path} 가 없음. 먼저 {SOURCE['where_is_signal.json']} 를 돌릴 것")
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    both = sig["D_무작위분할+누설"]
    rand = diag.loc["무작위분할_가중치없음_조기종료"]
    rows = [
        ("시간 분할 + 누설 배제 (최종)", leak.iloc[0]["PR_AUC"], leak.iloc[0]["포착률@1%"]),
        ("+ Id 순서 누설", leak.iloc[1]["PR_AUC"], leak.iloc[1]["포착률@1%"]),
        ("+ 무작위 분할", rand["PR_AUC"], rand["포착률@1%"]),
        ("+ 둘 다", both["PR_AUC"], both["포착률@1%"]),
    ]
    t = pd.DataFrame(rows, columns=["조건", "PR_AUC", "포착률@1%"])
    t["정직한_조건_대비"] = t["PR_AUC"] / t["PR_AUC"].iloc[0]
    return t


# ---------------------------------------------------------------- 대표 그림
####################################################################
# capture_curve — 검사 물량 ↔ 불량 포착률 (대표 그림)
#
# 읽는 법  대각선이 무작위 표본검사(검사율 = 포착률). 곡선이 대각선에서
#          위로 떨어진 만큼이 모델의 가치다. 오른쪽 축의 Lift 는 "같은
#          인원으로 무작위보다 몇 배 잡나".
# 가로축을 로그로 두는 이유  결론이 나는 자리가 0.5~3% 구간이라, 선형
#          축이면 그 구간이 왼쪽 끝에 뭉개진다.
####################################################################
def capture_curve(curve: pd.DataFrame, highlight: float | None = 0.01) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_min, x_max = curve["실제_검사율"].min(), curve["실제_검사율"].max()
    xs = np.geomspace(x_min, x_max, 100)
    ax.plot(xs * 100, xs * 100, ls="--", color=GREY, label="무작위 표본검사 (검사율 = 포착률)")
    styles = {"LightGBM": dict(color=MAIN, lw=2.2), "Logistic": dict(color="#DD8452", lw=1.4, ls="-.")}
    # 범례는 모델 이름이 아니라 역할을 앞에 둔다. 이 화면의 주인공은 모델이 아니라
    # "검사 물량을 얼마로 정할까" 라는 판단이다
    labels = {"LightGBM": "우리 모델 (LightGBM) 포착률", "Logistic": "비교 모델 (Logistic) 포착률"}
    for name, g in curve.groupby("모델", sort=False):
        ax.plot(g["실제_검사율"] * 100, g["포착률"] * 100,
                label=labels.get(name, f"{name} 포착률"), **styles.get(name, {}))
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("검사 물량 — 전체 생산품 중 추가로 보는 비율 (%, 로그 축)")
    ax.set_ylabel("불량 포착률 (%)")
    ax.grid(alpha=0.3, which="both")

    main = curve[curve["모델"] == "LightGBM"]
    ax2 = ax.twinx()
    ax2.plot(main["실제_검사율"] * 100, main["Lift"], color=MAIN, alpha=0.35, lw=1)
    ax2.set_ylabel("Lift — 무작위 대비 배수 (옅은 선)")
    ax2.set_ylim(bottom=0)

    if highlight is not None and len(main):
        row = main.iloc[(main["목표_검사율"] - highlight).abs().argmin()]
        ax.scatter([row["실제_검사율"] * 100], [row["포착률"] * 100], color=PICK, zorder=5)
        ax.annotate(
            f"{row['실제_검사율']:.1%} 검사 → 불량 {row['포착률']:.1%} 포착\n"
            f"({int(row['포착_불량'])}건, 무작위의 {row['Lift']:.1f}배)",
            (row["실제_검사율"] * 100, row["포착률"] * 100),
            xytext=(12, -38), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="-", color=PICK),
        )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("검사 물량 대비 불량 포착률 — 시험 구간(가장 늦은 20%)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- W2~W4
def inflation(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [MAIN] + ["#C44E52"] * (len(table) - 1)
    bars = ax.barh(table["조건"][::-1], table["정직한_조건_대비"][::-1], color=colors[::-1])
    for b, v, p in zip(bars, table["정직한_조건_대비"][::-1], table["PR_AUC"][::-1]):
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                f"{v:.1f}배 (PR-AUC {p:.3f})", va="center", fontsize=9)
    ax.set_xlabel("정직한 조건 대비 PR-AUC 배수")
    ax.set_xlim(0, table["정직한_조건_대비"].max() * 1.35)
    ax.set_title("평가 조건을 잘못 잡으면 성능이 몇 배 부풀려지나")
    fig.tight_layout()
    return fig


def early_prediction(curve: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["포함_공정수"], curve["Lift@1%"], marker="o", color=MAIN)
    ax.axhline(1, ls="--", color=GREY, label="무작위 (Lift 1)")
    # 뒤쪽 세 점(47·49·52공정)이 붙어 있어 이름을 왼쪽·위·아래로 나눠 단다
    offsets = [(6, 8), (6, 8), (-62, -4), (-14, 9), (-60, -16)]
    for (_, r), off in zip(curve.iterrows(), offsets + [(6, 8)] * len(curve)):
        ax.annotate(r["예측시점"], (r["포함_공정수"], r["Lift@1%"]),
                    xytext=off, textcoords="offset points", fontsize=8)
    ax.set_xlabel("예측 시점까지 거친 공정 수 (52개 중)")
    ax.set_ylabel("Lift@1% — 무작위 대비 배수")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=9)
    ax.set_title("예측 시점을 앞당기면 — 완만하게 줄지 않고 절벽처럼 떨어짐")
    fig.tight_layout()
    return fig


def threshold_stability(stab: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(stab))
    w = 0.27
    ax.bar(x - w, stab["목표_검사율"] * 100, w, color=GREY, label="목표 검사 물량")
    ax.bar(x, stab["고정_실제검사율"] * 100, w, color="#C44E52", label="확률 임계값 고정 (검증에서 정함)")
    ax.bar(x + w, stab["이동분위수_실제검사율"] * 100, w, color=MAIN, label="분위수로 다시 맞춤")
    for i, m in enumerate(stab["고정_검사율_오차배수"]):
        ax.text(i, stab["고정_실제검사율"].iloc[i] * 100, f"{m:.1f}배", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"목표 {t:.0%}" for t in stab["목표_검사율"]])
    ax.set_ylabel("시험 구간 실제 검사 물량 (%)")
    ax.legend(fontsize=8)
    ax.set_title("확률 임계값을 고정하면 기간이 바뀔 때 검사 물량이 2배 가까이 넘침")
    fig.tight_layout()
    return fig


def cost_ratio(cost: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(cost["비용비_r"], cost["최적_검사율"] * 100, where="post", color=MAIN, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    for _, r in cost.iterrows():
        ax.annotate(f"{r['최적_검사율']:.1%}", (r["비용비_r"], r["최적_검사율"] * 100),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("비용비 r = 불량 하나를 놓친 비용 ÷ 제품 하나를 검사하는 비용 (로그)")
    ax.set_ylabel("총비용이 가장 낮은 검사 물량 (%, 로그)")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("놓친 불량이 비쌀수록 — 어디서부터 전수 검사가 나은가")
    fig.text(0.01, 0.01, "검사 물량 후보 0.5·1·3·5·10·100% 중에서 고름. 꺾이는 지점은 후보 간격만큼 거침",
             fontsize=7, color=GREY)
    fig.tight_layout()
    return fig


def ablation(abl: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    t = abl.iloc[::-1]
    colors = ["#C44E52" if v < t["포착건수"].max() * 0.5 else MAIN for v in t["포착건수"]]
    bars = ax.barh(t["조건"], t["포착건수"], color=colors)
    for b, v in zip(bars, t["포착건수"]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{int(v)}건", va="center", fontsize=9)
    ax.set_xlabel("검사율 1%에서 잡은 불량 수 (시험 구간 불량 964건 중)")
    ax.set_xlim(0, t["포착건수"].max() * 1.2)
    # 표 순서: 첫 행 = 전체, 마지막 행 = 두 묶음 동시 제거, 그 사이 = 한 묶음씩 제거.
    # 제목 숫자를 표에서 계산한다 — "그대로"라고 쓰면 절벽 구간 단독 제거(-19건)와 어긋남
    full = int(abl["포착건수"].iloc[0])
    single_drop = full - int(abl["포착건수"].iloc[1:-1].min())
    both = int(abl["포착건수"].iloc[-1])
    ax.set_title(f"한 묶음씩 빼면 최대 {single_drop}건 줄지만, 두 묶음을 함께 빼면 {full}→{both}건 "
                 "— 신호가 겹쳐 있음", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- W1 F1~F5
def f1_structure(structure: pd.DataFrame) -> plt.Figure:
    t = structure.sort_values(["line", "station"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(t["station_key"], t["측정값수"], color=[LINE_COLORS[l] for l in t["line"]])
    ax.set_yscale("log")
    ax.set_ylabel("측정값 수 (로그)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LINE_COLORS.values()]
    ax.legend(handles, [f"라인 {l}" for l in LINE_COLORS], fontsize=8)
    ax.yaxis.set_major_formatter(PLAIN)
    # 날짜만 있고 측정값이 없는 공정이 있어 이 표의 공정 수는 52개보다 적다
    ax.set_title(f"F1 측정값이 있는 공정 {len(t)}개(전체 52개 중) — 측정값이 일부 공정에 몰려 있음")
    fig.tight_layout()
    return fig


def f2_missing_by_path(visit: pd.DataFrame) -> plt.Figure:
    t = visit.set_index(visit.columns[0])
    stations = [c for c in t.columns if c.startswith("L")]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(t[stations].to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(stations)), stations, rotation=90, fontsize=7)
    ax.set_yticks(range(len(t)), [f"{i} ({n:,}대)" for i, n in zip(t.index, t["제품수"])], fontsize=8)
    fig.colorbar(im, ax=ax, label="그 공정을 거친 제품 비율")
    ax.set_title("F2 결측은 무작위가 아니라 경로로 정해짐 — 상위 경로는 0 아니면 1")
    fig.tight_layout()
    return fig


####################################################################
# f3_paths — 상위 경로별 제품 수·불량률
#
# F2·위험 모니터와 같은 표(경로 묶음 = 학습 구간 기준 순위)를 쓴다.
# 처음엔 profile_data 의 T3_경로(전체 구간 기준 순위)를 썼는데, 그러면
# 같은 "경로 03" 이 F2 와 F3 에서 다른 경로를 가리켰음 (독립 검토 지적)
####################################################################
def f3_paths(visit: pd.DataFrame) -> plt.Figure:
    t = visit.set_index(visit.columns[0])
    t = t[~t.index.str.startswith("기타")]
    stations = [c for c in t.columns if c.startswith("L")]
    n_st = t[stations].sum(axis=1).round().astype(int)
    labels = [f"{i} ({n}공정)" for i, n in zip(t.index, n_st)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], t["제품수"][::-1], color=MAIN, alpha=0.7)
    ax.set_xlabel("제품 수 (막대)")
    ax2 = ax.twiny()
    ax2.plot(t["불량률"][::-1] * 100, labels[::-1], "o", color="#C44E52")
    ax2.set_xlabel("불량률 % (빨간 점)", color="#C44E52")
    ax.set_title(f"F3 상위 {len(t)}개 경로 — 경로마다 불량률이 다름 (번호는 F2 와 같음)", pad=30)
    fig.tight_layout()
    return fig


def f4_start_time(hist: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    width = (hist["구간_끝"] - hist["구간_시작"]).to_numpy()
    ax.bar(hist["구간_시작"], hist["제품수"], width=width, align="edge", color=MAIN, alpha=0.6)
    ax.set_xlabel("시작 시각 (익명 시간 단위)")
    ax.set_ylabel("제품 수 (막대)")
    ax2 = ax.twinx()
    mid = (hist["구간_시작"] + hist["구간_끝"]) / 2
    ax2.plot(mid, hist["불량률"] * 100, color="#C44E52")
    ax2.set_ylabel("불량률 % (빨간 선)", color="#C44E52")
    for col, name in (("검증_시작", "검증 시작"), ("시험_시작", "시험 시작")):
        x = hist[col].iloc[0]
        ax.axvline(x, color="k", ls=":")
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {name}", fontsize=9)
    ax.set_title("생산 기간과 불량률 — 기간마다 공정 상태가 다름 (시간 분할의 근거)")
    fig.tight_layout()
    return fig


def f5_order(order: pd.DataFrame) -> plt.Figure:
    t = order.sort_values("추정_공정순서")
    lines = t["station_key"].str.extract(r"L(\d)")[0].astype(int)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(range(len(t)), t["상대시각_중앙값"], c=[LINE_COLORS[l] for l in lines], s=25)
    ax.set_xticks(range(len(t)), t["station_key"], rotation=90, fontsize=7)
    ax.set_ylabel("제품 첫 공정 이후 경과 시각 중앙값")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c) for c in LINE_COLORS.values()]
    ax.legend(handles, [f"라인 {l}" for l in LINE_COLORS], fontsize=8)
    ax.set_title("F5 실제 공정 순서 추정 — 라인 번호 순서가 아님 (L0·L1 → L3 → L2)")
    fig.tight_layout()
    return fig


####################################################################
# build_all — 그림 파일명과 (그림 함수, 표) 를 짝지음
#
# 역할  main 과 테스트가 같은 목록을 쓴다. 그림 하나를 추가할 때
#       여기 한 줄만 늘리면 된다.
####################################################################
def build_all(report_dir: Path = REPORTS, run_dir: Path = RUNS) -> dict[str, plt.Figure]:
    L = lambda n, **kw: load(n, report_dir, **kw)  # noqa: E731
    return {
        "00_검사물량_포착률": capture_curve(L("T_포착률곡선")),
        "01_성능_부풀림": inflation(inflation_table(report_dir, run_dir)),
        "02_조기예측_곡선": early_prediction(L("T_조기예측곡선")),
        "03_임계값_안정성": threshold_stability(L("T_임계값안정성")),
        "04_비용비": cost_ratio(L("T_비용비곡선")),
        "05_ablation": ablation(L("T_ablation")),
        "F1_라인공정_구조": f1_structure(L("T2_구조")),
        "F2_경로별_결측구조": f2_missing_by_path(L("T_경로x공정_방문비율")),
        "F3_상위경로_불량률": f3_paths(L("T_경로x공정_방문비율")),
        "F4_시작시각_분포": f4_start_time(L("T_시작시각_분포")),
        "F5_공정순서_추정": f5_order(L("T_공정순서추정")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="집계표에서 그림 PNG 생성")
    ap.add_argument("--report-dir", default=REPORTS, type=Path)
    ap.add_argument("--out-dir", default=REPORTS / "figures", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        figs = build_all(args.report_dir)
    except FileNotFoundError as e:
        print(f"[중단] {e}")
        return 1
    for name, fig in figs.items():
        fig.savefig(args.out_dir / f"{name}.png", dpi=130)
        plt.close(fig)
    print(f"그림 {len(figs)}장을 {args.out_dir}/ 에 저장함")
    return 0


if __name__ == "__main__":
    sys.exit(main())


####################################################################
# 작업 메모
#   - 2026-10-19 작성. F2 는 제품 단위가 아니라 경로 단위 방문 비율로
#     그림 (제품 행을 그림으로 옮기지 않기 위함)
####################################################################
