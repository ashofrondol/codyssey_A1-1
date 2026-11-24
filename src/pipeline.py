"""이커머스 데이터 분석 파이프라인: DataAnalyzer 클래스.

원본 데이터를 읽어 결측치·이상치 처리, 멀티 모달 피처 엔지니어링(이미지·텍스트), 요약 통계,
RFM 고객 세분화를 수행한다. 같은 클래스로 거래 데이터와 상품 데이터를 모두 다룬다.

허용 라이브러리(NumPy, Pandas, Matplotlib)만 사용한다.
- 이미지는 명세가 허용한 `matplotlib.pyplot.imread()`로 파일을 배열로 읽기만 하고,
  흑백 변환·크기 맞추기·통계는 모두 NumPy 배열 연산으로 직접 한다.

사용 예:
    from pipeline import DataAnalyzer
    tx = DataAnalyzer("data/online_retail_II.csv")
    tx.load_data(parse_dates=["InvoiceDate"])
    tx.handle_missing_values("Description", strategy="group_mode", group_col="StockCode")
    outliers = tx.detect_outliers("Quantity", threshold=1.5)
    rfm = tx.calculate_rfm("Customer ID", "InvoiceDate", "amount")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 흑백 변환 가중치 (ITU-R BT.601): 사람 눈이 초록에 가장 민감하다는 점을 반영한 표준 값
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])

# RFM 세그먼트 규칙: 위에서부터 먼저 맞는 규칙을 적용한다. (R, F, M 점수는 1~5)
DEFAULT_SEGMENT_RULES = [
    ("VIP", lambda r, f, m: (r >= 4) & (f >= 4) & (m >= 4)),
    ("Loyal", lambda r, f, m: (r >= 3) & (f >= 3)),
    ("New", lambda r, f, m: (r >= 4) & (f <= 2)),
    ("At Risk", lambda r, f, m: (r <= 2) & (f >= 3)),
    ("Churned", lambda r, f, m: (r <= 2) & (f <= 2)),
]
DEFAULT_SEGMENT = "Potential"

# 상품이 아닌 거래 코드 (Online Retail II). 상품 코드는 숫자 5자리로 시작하고,
# DCGS…(반려동물·파티 용품), PADS(쿠션 패드), SP1002(어린이 이젤)도 실제 상품이다.
# 아래 코드는 상품 판매가 아니라 배송비·수수료·회계 조정 등이라 상품·고객 구매 분석에서 뺀다.
NON_PRODUCT_CODES = {
    "POST": "우편료", "DOT": "온라인 주문 우편료", "C2": "운송비", "C3": "운송비",
    "M": "수동 입력(Manual)", "D": "할인", "S": "샘플", "B": "악성 채권 조정",
    "ADJUST": "회계 조정", "ADJUST2": "회계 조정", "BANK CHARGES": "은행 수수료",
    "AMAZONFEE": "아마존 수수료", "CRUK": "자선단체 수수료", "GIFT": "상품권",
    "TEST001": "테스트 상품", "TEST002": "테스트 상품",
}
NON_PRODUCT_PREFIXES = {"GIFT_0001_": "상품권"}


class DataAnalyzer:
    """데이터 로드부터 RFM 세분화까지 한 흐름으로 처리하는 분석기.

    Attributes:
        data_path: 원본 CSV 경로.
        df: 현재 작업 중인 DataFrame. 각 메서드가 이 값을 읽고 고친다.
        log: 각 처리 단계에서 무엇을 얼마나 바꿨는지 기록한 리스트.
    """

    def __init__(self, data_path=None, df=None):
        self.data_path = Path(data_path) if data_path is not None else None
        self.df = df.copy() if df is not None else None
        self.log = []

    # ------------------------------------------------------------------
    # 1. 로드와 기본 탐색
    # ------------------------------------------------------------------

    def load_data(self, parse_dates=None, dtype=None, **read_kwargs):
        """CSV를 읽어 self.df에 저장한다.

        Args:
            parse_dates: 날짜형으로 읽을 컬럼 목록.
            dtype: 컬럼별 자료형 지정 (예: 코드값을 문자열로 읽기).
            **read_kwargs: pandas.read_csv에 그대로 전달할 인자.

        Returns:
            읽은 DataFrame.
        """
        if self.data_path is None:
            raise ValueError("data_path가 없다. DataAnalyzer(data_path=...)로 만들거나 df를 넘긴다.")
        self.df = pd.read_csv(self.data_path, parse_dates=parse_dates, dtype=dtype, **read_kwargs)
        self.log.append(f"load_data: {len(self.df):,}행 × {self.df.shape[1]}열")
        return self.df

    def missing_report(self):
        """컬럼별 결측치 개수와 비율(%)을 돌려준다."""
        n = self.df.isna().sum()
        return pd.DataFrame({"결측 개수": n, "결측 비율(%)": (n / len(self.df) * 100).round(2)})

    # ------------------------------------------------------------------
    # 1-1. 규칙 위반 정리 (중복, 비상품 코드)
    # ------------------------------------------------------------------

    def remove_duplicates(self, subset=None):
        """모든 열(또는 subset 열)이 같은 행을 하나만 남기고 지운다.

        Returns:
            지운 행의 수.
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset).copy()
        n = before - len(self.df)
        self.log.append(f"remove_duplicates: {n:,}행 삭제")
        return n

    def exclude_codes(self, code_col, codes=None, prefixes=None):
        """상품이 아닌 코드(배송비, 수수료, 조정 등)의 행을 뺀다.

        Args:
            code_col: 코드 컬럼 (예: StockCode).
            codes: {코드: 이유} 사전. 대소문자를 가리지 않는다. None이면 NON_PRODUCT_CODES.
            prefixes: {접두어: 이유} 사전. None이면 NON_PRODUCT_PREFIXES.

        Returns:
            뺀 행만 담은 DataFrame (reason 컬럼 포함).
        """
        codes = NON_PRODUCT_CODES if codes is None else codes
        prefixes = NON_PRODUCT_PREFIXES if prefixes is None else prefixes
        upper = self.df[code_col].astype(str).str.strip().str.upper()
        reason = upper.map({k.upper(): v for k, v in codes.items()})
        for pre, why in prefixes.items():
            reason = reason.where(~(reason.isna() & upper.str.startswith(pre.upper())), why)
        mask = reason.notna()
        removed = self.df[mask].assign(reason=reason[mask])
        self.df = self.df[~mask].copy()
        self.log.append(f"exclude_codes({code_col}): {int(mask.sum()):,}행 제외 (코드 {removed[code_col].nunique()}종)")
        return removed

    # ------------------------------------------------------------------
    # 2. 결측치 처리
    # ------------------------------------------------------------------

    def handle_missing_values(self, column, strategy="group_median", group_col=None, fill_value=None):
        """한 컬럼의 결측치를 지정한 전략으로 채운다.

        Args:
            column: 결측치를 채울 컬럼.
            strategy: 채우는 방법.
                - "group_mean", "group_median", "group_mode": group_col로 묶은 그룹의 평균/중앙값/최빈값.
                  그룹 안에 값이 하나도 없어 못 채운 칸은 그대로 결측으로 남는다.
                - "mean", "median", "mode": 컬럼 전체의 평균/중앙값/최빈값.
                - "constant": fill_value로 채운다.
                - "drop": 결측인 행을 지운다.
            group_col: 그룹 기준 컬럼 (group_* 전략에 필요).
            fill_value: constant 전략에서 쓸 값.

        Returns:
            채운(또는 지운) 칸의 수.
        """
        before = int(self.df[column].isna().sum())
        if strategy == "drop":
            self.df = self.df[self.df[column].notna()].copy()
            self.log.append(f"handle_missing_values({column}, drop): {before:,}행 삭제")
            return before

        if strategy.startswith("group_"):
            if group_col is None:
                raise ValueError(f"{strategy} 전략에는 group_col이 필요하다.")
            stat = strategy.removeprefix("group_")
            grouped = self.df.groupby(group_col)[column]
            if stat == "mode":
                fill = grouped.transform(_mode_or_nan)
            else:
                fill = grouped.transform(stat)
        elif strategy in ("mean", "median"):
            fill = getattr(self.df[column], strategy)()
        elif strategy == "mode":
            fill = _mode_or_nan(self.df[column])
        elif strategy == "constant":
            fill = fill_value
        else:
            raise ValueError(f"알 수 없는 strategy: {strategy}")

        self.df[column] = self.df[column].fillna(fill)
        filled = before - int(self.df[column].isna().sum())
        self.log.append(f"handle_missing_values({column}, {strategy}, group={group_col}): "
                        f"{filled:,}칸 채움, {before - filled:,}칸 남음")
        return filled

    # 명세의 메서드 이름은 handle_missing_values()다. 짧은 이름으로 불러도 같은 메서드가 실행되도록 별칭을 둔다.
    handle_missing = handle_missing_values

    # ------------------------------------------------------------------
    # 3. 이상치 (IQR)
    # ------------------------------------------------------------------

    def iqr_bounds(self, column, threshold=1.5):
        """IQR 방식의 이상치 경계를 계산한다.

        Q1(25%)과 Q3(75%) 사이 거리 IQR = Q3 - Q1 을 재고,
        [Q1 - threshold × IQR, Q3 + threshold × IQR] 밖을 이상치로 본다.

        Returns:
            dict(q1, q3, iqr, lower, upper).
        """
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        return {"q1": q1, "q3": q3, "iqr": iqr,
                "lower": q1 - threshold * iqr, "upper": q3 + threshold * iqr}

    def detect_outliers(self, column, threshold=1.5):
        """IQR 경계 밖에 있는 행을 돌려준다.

        Args:
            column: 검사할 수치형 컬럼.
            threshold: IQR에 곱할 배수. 1.5가 일반적이고, 3이면 아주 극단적인 값만 잡는다.

        Returns:
            이상치 행만 담은 DataFrame.
        """
        b = self.iqr_bounds(column, threshold)
        mask = (self.df[column] < b["lower"]) | (self.df[column] > b["upper"])
        return self.df[mask]

    def treat_outliers(self, column, threshold=1.5, method="clip"):
        """IQR 이상치를 처리한다.

        Args:
            method: "clip"이면 경계값으로 잘라 붙이고(행 유지), "remove"면 행을 지운다.

        Returns:
            처리한 행의 수.
        """
        b = self.iqr_bounds(column, threshold)
        mask = (self.df[column] < b["lower"]) | (self.df[column] > b["upper"])
        n = int(mask.sum())
        if method == "clip":
            self.df[column] = self.df[column].clip(b["lower"], b["upper"])
        elif method == "remove":
            self.df = self.df[~mask].copy()
        else:
            raise ValueError(f"알 수 없는 method: {method}")
        self.log.append(f"treat_outliers({column}, threshold={threshold}, {method}): {n:,}행 "
                        f"(경계 {b['lower']:.2f} ~ {b['upper']:.2f})")
        return n

    # ------------------------------------------------------------------
    # 4. 멀티 모달 피처 엔지니어링
    # ------------------------------------------------------------------

    @staticmethod
    def load_images(paths, size=(80, 60), step=1):
        """이미지 파일들을 흑백 배열 하나로 쌓는다.

        `plt.imread()`로 파일을 배열로 읽은 뒤의 처리는 모두 NumPy로 한다.
        - 값 범위 맞추기: 정수(0~255)로 읽힌 JPG는 255로 나눠 0~1 실수로 맞춘다.
        - 흑백 변환: RGB에 GRAY_WEIGHTS를 곱해 더한다(행렬 곱).
        - 크기 맞추기: 가운데를 size만큼 잘라내고, 모자라면 흰색(1.0)으로 채운다.
        - 다운샘플링: arr[::step, ::step].

        Args:
            paths: 이미지 파일 경로 목록.
            size: (높이, 너비). 모든 이미지를 이 크기로 맞춘다.
            step: 다운샘플링 간격 (1이면 그대로).

        Returns:
            shape (N, 높이 // step, 너비 // step)의 float 배열.
        """
        h, w = size
        out = np.ones((len(paths), h, w), dtype=np.float32)
        for i, p in enumerate(paths):
            img = plt.imread(p)
            if np.issubdtype(img.dtype, np.integer):
                img = img / 255.0
            if img.ndim == 3:
                img = img[..., :3] @ GRAY_WEIGHTS
            ih, iw = img.shape
            top, left = max((ih - h) // 2, 0), max((iw - w) // 2, 0)
            crop = img[top:top + h, left:left + w]
            ch, cw = crop.shape
            oy, ox = (h - ch) // 2, (w - cw) // 2
            out[i, oy:oy + ch, ox:ox + cw] = crop
        return out[:, ::step, ::step]

    def add_image_features(self, images, prefix="img"):
        """이미지마다 밝기 평균과 표준편차를 계산해 컬럼으로 추가한다.

        반복문 없이 축(axis)을 지정한 배열 연산으로 N장을 한 번에 계산한다.
        - 평균: 이미지가 전체적으로 얼마나 밝은가 (1에 가까울수록 흰 배경이 많음)
        - 표준편차: 밝기가 얼마나 들쭉날쭉한가 (대비, 무늬가 많을수록 큼)

        Args:
            images: shape (N, H, W) 배열. self.df의 행 순서와 같아야 한다.
            prefix: 새 컬럼 이름 앞부분.
        """
        if len(images) != len(self.df):
            raise ValueError(f"이미지 {len(images)}장과 행 {len(self.df)}개의 수가 다르다.")
        self.df[f"{prefix}_mean"] = images.mean(axis=(1, 2))
        self.df[f"{prefix}_std"] = images.std(axis=(1, 2))
        self.log.append(f"add_image_features: {len(images):,}장 → {prefix}_mean, {prefix}_std")

    def add_text_features(self, column, prefix=None):
        """텍스트 컬럼에서 단어 수와 글자 수를 계산해 컬럼으로 추가한다."""
        prefix = prefix or column
        text = self.df[column].fillna("").astype(str).str.strip()
        self.df[f"{prefix}_word_count"] = text.str.split().str.len()
        self.df[f"{prefix}_char_length"] = text.str.len()
        self.log.append(f"add_text_features({column}) → {prefix}_word_count, {prefix}_char_length")

    @staticmethod
    def image_histogram_features(images, bins=8):
        """(보너스) 이미지마다 밝기 히스토그램을 계산한다. 각 구간에 속한 픽셀 비율.

        구간 번호를 배열 연산으로 구한 뒤, 이미지 번호와 합쳐 한 번의 bincount로 센다.

        Returns:
            shape (N, bins) 배열. 각 행의 합은 1.
        """
        n = len(images)
        flat = images.reshape(n, -1)
        idx = np.clip((flat * bins).astype(int), 0, bins - 1)
        counts = np.bincount((idx + np.arange(n)[:, None] * bins).ravel(), minlength=n * bins)
        return counts.reshape(n, bins) / flat.shape[1]

    @staticmethod
    def edge_strength(images):
        """(보너스) 이미지마다 평균 엣지 세기를 계산한다.

        옆 픽셀, 아래 픽셀과의 밝기 차이(np.diff)로 기울기를 구하고, 그 크기
        sqrt(dx² + dy²)의 평균을 낸다. 무늬나 윤곽이 많을수록 크다.

        Returns:
            shape (N,) 배열.
        """
        dx = np.diff(images, axis=2)[:, :-1, :]
        dy = np.diff(images, axis=1)[:, :, :-1]
        return np.sqrt(dx ** 2 + dy ** 2).mean(axis=(1, 2))

    # ------------------------------------------------------------------
    # 5. 통계
    # ------------------------------------------------------------------

    def summary_stats(self, columns):
        """평균, 중앙값, 표준편차, 사분위수(Q1, Q3)를 한 표로 돌려준다."""
        d = self.df[columns]
        return pd.DataFrame({
            "평균": d.mean(), "중앙값": d.median(), "표준편차": d.std(),
            "Q1(25%)": d.quantile(0.25), "Q3(75%)": d.quantile(0.75),
            "최솟값": d.min(), "최댓값": d.max(),
        })

    def correlations(self, pairs, method="pearson"):
        """변수 쌍마다 상관계수를 계산한다.

        Args:
            pairs: [(컬럼1, 컬럼2), ...].
            method: "pearson"(직선 관계) 또는 "spearman"(순위 관계, 극단값에 덜 민감).
                spearman은 두 컬럼을 각각 순위로 바꾼 뒤 그 순위끼리의 pearson 상관계수로 직접 계산한다
                (pandas의 spearman은 허용 라이브러리가 아닌 SciPy를 필요로 하기 때문).
        """
        if method not in ("pearson", "spearman"):
            raise ValueError(f"알 수 없는 method: {method}")
        rows = []
        for a, b in pairs:
            x, y = self.df[a], self.df[b]
            if method == "spearman":
                x, y = x.rank(), y.rank()
            rows.append((a, b, x.corr(y)))
        return pd.DataFrame(rows, columns=["변수1", "변수2", f"상관계수({method})"])

    # ------------------------------------------------------------------
    # 6. RFM
    # ------------------------------------------------------------------

    def calculate_rfm(self, customer_col, date_col, amount_col, invoice_col=None,
                      reference_date=None, n_bins=5, segment_rules=None,
                      cancel_prefix=None, drop_nonpositive=True):
        """고객별 Recency, Frequency, Monetary와 점수, 세그먼트를 계산한다.

        - Recency(최근성): 기준일 - 마지막 구매일 (일). 작을수록 좋다.
        - Frequency(빈도): 구매 횟수. invoice_col이 있으면 서로 다른 주문 수, 없으면 행 수.
        - Monetary(금액): 총 구매 금액. cancel_prefix를 주면 취소 금액(음수)을 뺀 순 구매 금액이다.
        점수는 고객 순위를 n_bins개 구간으로 나눠 1~n_bins를 준다. 높을수록 좋은 고객이다
        (Recency는 작을수록 높은 점수).

        Args:
            customer_col, date_col, amount_col: 고객 ID, 거래일, 금액 컬럼.
            invoice_col: 주문 번호 컬럼 (선택).
            reference_date: 기준일. None이면 데이터의 최종 거래일 다음 날.
            n_bins: 점수 구간 수.
            segment_rules: [(이름, 함수(r, f, m) → bool Series), ...]. None이면 기본 규칙.
            cancel_prefix: 취소 주문 번호의 접두어(예: "C"). 주면 취소 행은 Recency·Frequency 계산에서
                빼고, Monetary에는 음수 금액으로 더해 순 구매 금액을 만든다. invoice_col이 필요하다.
            drop_nonpositive: 순 구매 금액이 0 이하인 고객(취소가 구매보다 많은 고객)을 뺄지 여부.

        Returns:
            고객별 RFM DataFrame (Recency, Frequency, Monetary, R, F, M, RFM_score, segment).
        """
        d = self.df.dropna(subset=[customer_col])
        if cancel_prefix is not None:
            if invoice_col is None:
                raise ValueError("cancel_prefix를 쓰려면 invoice_col이 필요하다.")
            buy = d[~d[invoice_col].astype(str).str.startswith(cancel_prefix)]
        else:
            buy = d
        if reference_date is None:
            reference_date = buy[date_col].max().normalize() + pd.Timedelta(days=1)
        reference_date = pd.Timestamp(reference_date)
        freq = (buy.groupby(customer_col)[invoice_col].nunique() if invoice_col
                else buy.groupby(customer_col).size())
        rfm = pd.DataFrame({
            "Recency": (reference_date - buy.groupby(customer_col)[date_col].max()).dt.days,
            "Frequency": freq,
        })
        rfm["Monetary"] = d.groupby(customer_col)[amount_col].sum().reindex(rfm.index)
        n_drop = 0
        if drop_nonpositive:
            keep = rfm["Monetary"] > 0
            n_drop = int((~keep).sum())
            rfm = rfm[keep].copy()
        labels = list(range(1, n_bins + 1))
        # 같은 값이 많아도 구간이 깨지지 않도록 순위(rank)를 매긴 뒤 등분한다
        rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first", ascending=False), n_bins, labels=labels).astype(int)
        rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), n_bins, labels=labels).astype(int)
        rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), n_bins, labels=labels).astype(int)
        rfm["RFM_score"] = rfm["R"] + rfm["F"] + rfm["M"]
        rfm["segment"] = self.segment_rfm(rfm, segment_rules)
        self.log.append(f"calculate_rfm: 고객 {len(rfm):,}명, 기준일 {reference_date.date()}"
                        + (f", 순 구매 금액 0 이하 {n_drop}명 제외" if n_drop else ""))
        return rfm

    @staticmethod
    def segment_rfm(rfm, rules=None, default=DEFAULT_SEGMENT):
        """R, F, M 점수에 규칙을 위에서부터 적용해 세그먼트 이름을 붙인다."""
        rules = rules or DEFAULT_SEGMENT_RULES
        seg = pd.Series(default, index=rfm.index, dtype=object)
        assigned = pd.Series(False, index=rfm.index)
        for name, rule in rules:
            hit = rule(rfm["R"], rfm["F"], rfm["M"]) & ~assigned
            seg[hit] = name
            assigned |= hit
        return seg

    def cohort_retention(self, customer_col, date_col, freq="M"):
        """(보너스) 첫 구매 월별 코호트의 월별 재구매율(Retention Rate)을 계산한다.

        가입일이 없으므로 고객의 첫 구매 월을 가입 월로 본다.
        재구매율 = (코호트 고객 중 k개월 뒤에도 구매한 고객 수) / (코호트 전체 고객 수).

        Returns:
            행 = 코호트(첫 구매 월), 열 = 경과 개월 수(0, 1, 2, ...)인 비율 표.
            데이터 기간 밖이라 관측할 수 없는 칸은 NaN이다.
        """
        d = self.df.dropna(subset=[customer_col])[[customer_col, date_col]].copy()
        d["order_month"] = d[date_col].dt.to_period(freq)
        d["cohort"] = d.groupby(customer_col)["order_month"].transform("min")
        d["months_since"] = (d["order_month"] - d["cohort"]).apply(lambda x: x.n)
        counts = d.groupby(["cohort", "months_since"])[customer_col].nunique().unstack(fill_value=0)
        rate = counts.div(counts[0], axis=0)
        # 데이터 끝을 넘어선 칸(예: 마지막 달 코호트의 3개월 뒤)은 아직 관측할 수 없으므로 0이 아니라 NaN으로 둔다
        last = d["order_month"].max()
        observable = np.array([[m <= (last - c).n for m in rate.columns] for c in rate.index])
        return rate.where(observable)


def _mode_or_nan(s):
    """최빈값 하나를 돌려준다. 값이 없으면 NaN."""
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan
