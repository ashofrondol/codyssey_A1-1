"""과제 데이터를 내려받아 data/ 에 준비한다. 표준 라이브러리만 사용한다.

1. UCI Online Retail II (CC BY 4.0)
   - zip 안의 엑셀(xlsx)을 openpyxl 없이 zipfile + XML 파싱으로 읽어 CSV로 바꾼다.
     (xlsx는 XML 파일들을 zip으로 묶은 형식이다. 허용 라이브러리 밖의 패키지를 쓰지 않기 위해 직접 읽는다.)
   - 결과: data/online_retail_II.csv (두 시트를 합친 1,067,371건)
2. Fashion Product Images (Small) (MIT, 원본: Kaggle paramaggarwal)
   - 원본 Kaggle은 로그인이 필요해서 같은 데이터의 Hugging Face 미러(ashraq/fashion-product-images-small)를
     datasets-server API로 받는다.
   - 결과: data/products.csv (상품 메타데이터), data/images/<id>.jpg (상품 이미지)

실행:
    uv run python scripts/download_data.py            # 상품 2,000개
    uv run python scripts/download_data.py --n 3000   # 상품 개수 변경
"""

import argparse
import csv
import io
import json
import random
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree.ElementTree import iterparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
HF_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=ashraq/fashion-product-images-small"
           "&config=default&split=train&offset={offset}&length={length}")
HF_TOTAL = 44072
SEED = 42

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EXCEL_EPOCH = datetime(1899, 12, 30)


def fetch(url, retries=8, pause=0.05):
    """URL의 내용을 bytes로 받는다.

    서버가 잠시 막으면(503 등) 2, 4, 8, ... 초(최대 60초)씩 기다렸다가 다시 시도한다.
    요청마다 pause초 쉬어 서버에 부담을 덜 준다.
    """
    for i in range(retries):
        try:
            time.sleep(pause)
            req = urllib.request.Request(url, headers={"User-Agent": "codyssey-a1-1-data-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(min(2 ** (i + 1), 60))


# ---------------------------------------------------------------------------
# 1. Online Retail II: xlsx → csv
# ---------------------------------------------------------------------------

def col_index(ref):
    """셀 주소의 열 부분(예: 'C12' → 2)을 0부터 시작하는 번호로 바꾼다."""
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return n - 1


def read_shared_strings(zf):
    """xlsx의 공유 문자열 표(sharedStrings.xml)를 리스트로 읽는다."""
    strings = []
    with zf.open("xl/sharedStrings.xml") as f:
        for _, el in iterparse(f):
            if el.tag == NS + "si":
                strings.append("".join(t.text or "" for t in el.iter(NS + "t")))
                el.clear()
    return strings


def iter_sheet_rows(zf, path, strings):
    """시트 XML을 한 행씩 읽어 값 리스트를 돌려준다."""
    with zf.open(path) as f:
        for _, el in iterparse(f):
            if el.tag != NS + "row":
                continue
            values = {}
            for c in el.iter(NS + "c"):
                t = c.get("t")
                v = c.find(NS + "v")
                if t == "s":
                    val = strings[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(NS + "t"))
                else:
                    val = v.text if v is not None else ""
                values[col_index(c.get("r"))] = val
            el.clear()
            if values:
                yield [values.get(i, "") for i in range(max(values) + 1)]


def clean_number_text(x):
    """엑셀이 숫자로 저장한 코드값(예: '85048.0')을 '85048'로 바꾼다."""
    if x.endswith(".0") and x[:-2].lstrip("-").isdigit():
        return x[:-2]
    return x


def excel_serial_to_text(x):
    """엑셀 날짜 일련번호(1900 날짜 체계)를 'YYYY-MM-DD HH:MM:SS'로 바꾼다."""
    if not x:
        return ""
    dt = EXCEL_EPOCH + timedelta(days=float(x))
    return (dt + timedelta(seconds=0.5)).strftime("%Y-%m-%d %H:%M:%S")


def prepare_online_retail():
    out = DATA / "online_retail_II.csv"
    if out.exists():
        print(f"[건너뜀] {out.relative_to(ROOT)} 이미 있음")
        return
    zpath = RAW / "online_retail_II.zip"
    if not zpath.exists():
        print("[받는 중] UCI Online Retail II zip")
        zpath.write_bytes(fetch(UCI_URL))
    with zipfile.ZipFile(zpath) as outer:
        xlsx_name = next(n for n in outer.namelist() if n.endswith(".xlsx"))
        xlsx = zipfile.ZipFile(io.BytesIO(outer.read(xlsx_name)))
    strings = read_shared_strings(xlsx)
    sheets = sorted(n for n in xlsx.namelist() if n.startswith("xl/worksheets/sheet"))
    n_rows = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = None
        for sheet in sheets:
            rows = iter_sheet_rows(xlsx, sheet, strings)
            head = next(rows)
            if header is None:
                header = head
                w.writerow(header)
                i_date = header.index("InvoiceDate")
                i_text = [header.index(c) for c in ("Invoice", "StockCode", "Customer ID")]
            for r in rows:
                r = r + [""] * (len(header) - len(r))
                r[i_date] = excel_serial_to_text(r[i_date])
                for i in i_text:
                    r[i] = clean_number_text(r[i])
                w.writerow(r)
                n_rows += 1
            print(f"  {sheet}: 누적 {n_rows:,}건")
    print(f"[저장] {out.relative_to(ROOT)} ({n_rows:,}건)")


# ---------------------------------------------------------------------------
# 2. Fashion Product Images (Small): 메타데이터 + 이미지
# ---------------------------------------------------------------------------

def prepare_products(n):
    out = DATA / "products.csv"
    img_dir = DATA / "images"
    if out.exists():
        print(f"[건너뜀] {out.relative_to(ROOT)} 이미 있음")
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    block = 100
    offsets = sorted(rng.sample(range(0, HF_TOTAL - block, block), n // block))
    fields = ["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour",
              "season", "year", "usage", "productDisplayName", "image_path"]
    rows = []
    for k, off in enumerate(offsets, 1):
        data = json.loads(fetch(HF_ROWS.format(offset=off, length=block)))
        for item in data["rows"]:
            r = item["row"]
            path = img_dir / f"{r['id']}.jpg"
            if not path.exists():
                path.write_bytes(fetch(r["image"]["src"]))
            rows.append({**{f: r.get(f) for f in fields[:-1]},
                         "image_path": str(path.relative_to(ROOT))})
        print(f"  묶음 {k}/{len(offsets)} (offset {off}) 누적 {len(rows):,}개")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[저장] {out.relative_to(ROOT)} ({len(rows):,}개), 이미지 {img_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="받을 상품 수 (100 단위)")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    prepare_online_retail()
    prepare_products(args.n)
