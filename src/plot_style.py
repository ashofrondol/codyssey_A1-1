"""그림 공통 설정: 한글 글꼴, 저장 위치(outputs/)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

BLUE, RED, GREEN, ORANGE, GRAY, PURPLE = "#378ADD", "#E24B4A", "#1D9E75", "#BA7517", "#888780", "#7F77DD"
SEGMENT_COLORS = {"VIP": "#7F77DD", "Loyal": "#1D9E75", "Potential": "#378ADD",
                  "New": "#BA7517", "At Risk": "#E24B4A", "Churned": "#888780"}


def setup_style():
    """설치된 한글 글꼴을 골라 matplotlib 기본값을 맞춘다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("AppleGothic", "NanumGothic", "Malgun Gothic") if f in installed),
                "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font,
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",   # 로그 축 눈금(10^-1 등)의 기호를 DejaVu로 그린다
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plain_log_ticks(ax, axis="y"):
    """로그 축 눈금을 10의 거듭제곱 수식 대신 일반 숫자(0.001, 10, 1e+05 등)로 쓴다.

    수식 눈금은 한글 글꼴에 없는 빼기 기호(U+2212)를 써서 글자가 깨진다.
    """
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save_figure(fig, name):
    """그림을 outputs/<name>.png 로 저장한다."""
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=110)
    print(f"[저장] outputs/{name}.png")
    return path
