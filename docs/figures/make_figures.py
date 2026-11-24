"""README 개념 설명용 그림을 만든다.

과제 결과(outputs/)와 별개다. 원본 데이터 없이도 다시 만들 수 있도록
개념을 보여 주는 **가상의 값**(시드 42)만 사용한다.

실행:
    uv run python docs/figures/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent
_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": next((f for f in ("AppleGothic", "NanumGothic", "Malgun Gothic") if f in _installed), "DejaVu Sans"),
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, RED, GREEN, ORANGE, GRAY, PURPLE = "#378ADD", "#E24B4A", "#1D9E75", "#BA7517", "#888780", "#7F77DD"
rng = np.random.default_rng(42)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def spearman(x, y):
    """순위로 바꾼 뒤 Pearson 상관계수 (동점은 없다고 가정한 간단한 버전)."""
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def fig_mean_median():
    """오른쪽으로 긴 분포에서 평균과 중앙값이 갈라지는 모습, 로그 축으로 보면 대칭에 가까워지는 모습."""
    x = rng.lognormal(mean=2.3, sigma=1.0, size=20000)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.hist(x[x < 120], bins=80, color=GRAY)
    a1.axvline(np.median(x), color=RED, lw=2, label=f"중앙값 {np.median(x):.1f}")
    a1.axvline(x.mean(), color=ORANGE, lw=2, ls="--", label=f"평균 {x.mean():.1f}")
    a1.set(title="오른쪽으로 긴 분포: 큰 값 몇 개가 평균을 끌어올린다",
           xlabel="주문 금액 (가상, 120 미만만 표시)", ylabel="주문 수")
    a1.legend()
    a2.hist(np.log10(x), bins=80, color=BLUE)
    a2.axvline(np.log10(np.median(x)), color=RED, lw=2, label="중앙값")
    a2.axvline(np.log10(x.mean()), color=ORANGE, lw=2, ls="--", label="평균")
    a2.set_xticks([-1, 0, 1, 2, 3])
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10.0 ** v:g}"))
    a2.set(title="같은 데이터를 로그 축으로: 좌우가 비슷해진다", xlabel="주문 금액 (로그 축)", ylabel="주문 수")
    a2.legend()
    save(fig, "fig01_mean_median")


def fig_boxplot_anatomy():
    """박스플롯의 각 부분이 무엇을 뜻하는지."""
    x = np.r_[rng.normal(50, 10, 300), [95, 102, 110, 8]]
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.boxplot(x, orientation="horizontal", widths=0.5, whis=1.5, patch_artist=True,
               boxprops=dict(facecolor="#E6F1FB", edgecolor=BLUE),
               medianprops=dict(color=RED, lw=2), flierprops=dict(marker="o", markerfacecolor=ORANGE, markeredgecolor=ORANGE))
    ax.axvline(lo, color=GRAY, ls=":", lw=1)
    ax.axvline(hi, color=GRAY, ls=":", lw=1)
    for v, t, c, tx, ty in [(q1, f"Q1 (25%)\n{q1:.1f}", BLUE, q1 - 9, 1.45), (med, f"중앙값 (50%)\n{med:.1f}", RED, med, 1.6),
                            (q3, f"Q3 (75%)\n{q3:.1f}", BLUE, q3 + 9, 1.45)]:
        ax.annotate(t, (v, 1.25), (tx, ty), ha="center", color=c, fontsize=9, arrowprops=dict(arrowstyle="-", color=c))
    ax.annotate("", (q1, 0.62), (q3, 0.62), arrowprops=dict(arrowstyle="<->", color=PURPLE))
    ax.text(med, 0.55, f"IQR = Q3 - Q1 = {iqr:.1f}\n(가운데 50%가 퍼진 폭)", ha="center", va="top", color=PURPLE, fontsize=9)
    ax.text(lo, 0.72, f"하한 = Q1 - 1.5×IQR\n{lo:.1f}", ha="center", fontsize=8, color=GRAY)
    ax.text(hi, 0.72, f"상한 = Q3 + 1.5×IQR\n{hi:.1f}", ha="center", fontsize=8, color=GRAY)
    ax.text(x.max(), 1.13, "이상치\n(경계 밖 점)", ha="center", fontsize=9, color=ORANGE)
    ax.set(ylim=(0.25, 1.75), yticks=[], xlabel="값 (가상)",
           title="박스플롯 읽는 법: 상자 = 가운데 50%, 수염 = 경계 안의 가장 먼 값, 점 = 이상치")
    save(fig, "fig02_boxplot_anatomy")


def fig_correlation():
    """Pearson과 Spearman이 같은 데이터를 어떻게 다르게 보는지."""
    n = 200
    x1 = rng.normal(0, 1, n); y1 = x1 + rng.normal(0, 0.45, n)
    x2 = rng.normal(0, 1, n); y2 = rng.normal(0, 1, n)
    x3 = rng.uniform(0, 4, n); y3 = np.exp(x3) + rng.normal(0, 0.3, n)
    x4 = rng.normal(0, 1, n); y4 = rng.normal(0, 1, n)
    x4[:3] = [12, 13, 14]; y4[:3] = [12, 14, 13]
    cases = [(x1, y1, "직선 관계"), (x2, y2, "관계 없음"), (x3, y3, "곡선이지만 한 방향 (단조)"),
             (x4, y4, "관계 없음 + 극단값 3개")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, (x, y, t) in zip(axes, cases):
        ax.scatter(x, y, s=10, color=BLUE, alpha=0.6)
        ax.set(title=f"{t}\nPearson {np.corrcoef(x, y)[0, 1]:.2f} / Spearman {spearman(x, y):.2f}",
               xlabel="x", ylabel="y")
    fig.suptitle("상관계수: Pearson은 직선에 얼마나 가까운가, Spearman은 순위가 얼마나 같이 움직이는가", y=1.12)
    save(fig, "fig03_correlation")


def fig_image_as_array():
    """흑백 이미지는 0~1 숫자 표이고, 컬러는 RGB 세 장을 가중합해 흑백으로 바꾼다."""
    img = np.ones((8, 6))
    img[1:3, 0:6] = 0.2          # 소매
    img[1:7, 1:5] = 0.2          # 몸통
    img[1, 2:4] = 0.9            # 목
    img[3:5, 2:4] = 0.5          # 무늬
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.imshow(img, cmap="gray", vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(img):
        a1.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=9, color="black" if v > 0.5 else "white")
    a1.set(title=f"8×6 흑백 이미지 = 숫자 48개\n평균 {img.mean():.2f}, 표준편차 {img.std():.2f}",
           xticks=range(6), yticks=range(8), xlabel="열 (가로 위치)", ylabel="행 (세로 위치)")
    rgb = np.array([[0.9, 0.2, 0.2], [0.2, 0.7, 0.2], [0.2, 0.3, 0.9], [0.9, 0.9, 0.9]])
    w = np.array([0.299, 0.587, 0.114])
    gray = rgb @ w
    names = ["빨강", "초록", "파랑", "흰색"]
    for i, (c, g) in enumerate(zip(rgb, gray)):
        a2.add_patch(plt.Rectangle((0, 3 - i), 0.8, 0.8, color=c))
        a2.add_patch(plt.Rectangle((3.4, 3 - i), 0.8, 0.8, color=str(g)))
        a2.text(1.0, 3.4 - i, f"R {c[0]:.1f}, G {c[1]:.1f}, B {c[2]:.1f}", va="center", fontsize=9)
        a2.text(2.8, 3.4 - i, "→", va="center", fontsize=14)
        a2.text(4.4, 3.4 - i, f"{names[i]} → 밝기 {g:.2f}", va="center", fontsize=9)
    a2.text(0, 4.3, "흑백 밝기 = 0.299 × R + 0.587 × G + 0.114 × B", fontsize=11)
    a2.set(xlim=(0, 6.4), ylim=(-0.2, 4.8), xticks=[], yticks=[],
           title="컬러 → 흑백: 사람 눈이 초록에 가장 민감해 초록 가중치가 크다")
    for s in a2.spines.values():
        s.set_visible(False)
    save(fig, "fig04_image_as_array")


def fig_rfm_scoring():
    """RFM 점수: 고객을 순위대로 세운 뒤 5등분해 1~5점을 준다."""
    freq = np.sort(rng.integers(1, 40, 20))
    score = np.repeat(np.arange(1, 6), 4)
    colors = ["#F1EFE8", "#B5D4F4", "#85B7EB", "#378ADD", "#185FA5"]
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(20), freq, color=[colors[s - 1] for s in score], edgecolor=GRAY)
    for i, (b, s) in enumerate(zip(bars, score)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{s}점", ha="center", fontsize=8)
    for k in range(1, 5):
        ax.axvline(4 * k - 0.5, color=GRAY, ls=":", lw=1)
    ax.set(xticks=range(20), xticklabels=[f"고객{i + 1}" for i in range(20)], ylabel="주문 수 (Frequency)",
           xlabel="고객 (주문 수가 적은 순서)",
           title="F 점수 매기기: 20명을 주문 수 순서로 세우고 4명씩 5구간으로 나눠 1~5점 (R은 최근일수록 높은 점수)")
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    save(fig, "fig05_rfm_scoring")


def fig_pareto():
    """파레토 곡선: 많이 쓰는 고객부터 쌓으면 소수가 매출 대부분을 낸다."""
    spend = np.sort(rng.lognormal(mean=6.5, sigma=1.4, size=2000))[::-1]
    cust = np.arange(1, len(spend) + 1) / len(spend) * 100
    rev = np.cumsum(spend) / spend.sum() * 100
    at20 = rev[int(len(spend) * 0.2) - 1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(cust, rev, color=BLUE, lw=2.5, label="누적 매출 (가상 고객 2,000명)")
    ax.plot([0, 100], [0, 100], color=GRAY, ls="--", label="모든 고객이 똑같이 쓴다면")
    ax.plot([20, 20, 0], [0, at20, at20], color=RED, ls=":")
    ax.annotate(f"상위 20% 고객이\n매출의 {at20:.0f}%", (20, at20), (32, at20 - 22), color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set(xlim=(0, 100), ylim=(0, 102), xlabel="고객 비율 (%, 많이 쓴 고객부터)", ylabel="누적 매출 비율 (%)",
           title="파레토 곡선: 고객 매출이 고르지 않을수록 곡선이 왼쪽 위로 휜다")
    ax.legend(loc="lower right")
    save(fig, "fig06_pareto")


if __name__ == "__main__":
    fig_mean_median()
    fig_boxplot_anatomy()
    fig_correlation()
    fig_image_as_array()
    fig_rfm_scoring()
    fig_pareto()
