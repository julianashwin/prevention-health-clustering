"""Download every public PSID file and its documentation, then convert to parquet.

Three steps, each safe to re-run (finished files are skipped):

1. data:    logs in to the PSID Data Center with your own account and downloads
            every zip on the Packaged Data page (family files 1968-2023, the
            cross-year individual file, wealth files, histories, supplements).
            "Sensitive Data Variables" files need a separate agreement and are
            skipped unless you pass --include-sensitive.
2. docs:    downloads every PDF/XLSX/XLS/ZIP linked from the PSID documentation
            page (questionnaires, codebooks, QxQs, user guides, weights notes),
            plus the cross-year variable index.
3. convert: unzips each data zip and reads the fixed-width text with the SAS
            setup file shipped inside it, writing one parquet per data file and
            a CSV of variable labels.

Usage (from the repo root):

    .venv/bin/python data_cleaning/psid/00_download_psid.py

Options: --steps data,docs,convert (default all), --out DIR
(default data/raw/psid), --only 1214,1053 (Data Center file ids),
--include-sensitive, --supplements none|all (default all).

Your password is read with getpass and never written to disk. The PSID
conditions of use apply to everything downloaded; outputs live under data/,
which is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import html
import http.cookiejar
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

SIMBA = "https://simba.isr.umich.edu"
LOGIN_URL = f"{SIMBA}/U/Login.aspx"
ZIPMAIN_URL = f"{SIMBA}/Zips/ZipMain.aspx"
DOCS_URL = "https://psidonline.isr.umich.edu/Guide/documents.aspx"
EXTRA_DOCS = [
    # Cross-year variable index (one row per concept, one column per year).
    "https://psidonline.isr.umich.edu/help/xyr/psid.xlsx",
    "https://psidonline.isr.umich.edu/data/Documentation/UserGuide2023.pdf",
]
DOC_EXTENSIONS = (".pdf", ".xlsx", ".xls", ".zip", ".txt", ".doc", ".docx", ".csv")
USER_AGENT = "Mozilla/5.0 (research download script; PSID registered user)"

# Fallback if the Packaged Data page cannot be parsed: (file id, label).
# The live page is always tried first, so new releases are picked up.
FALLBACK_FILES = [
    *[(fid, f"Family {yr}") for yr, fid in [
        (1968, 1056), (1969, 1058), (1970, 1059), (1971, 1060), (1972, 1061),
        (1973, 1062), (1974, 1063), (1975, 1064), (1976, 1065), (1977, 1066),
        (1978, 1067), (1979, 1068), (1980, 1069), (1981, 1070), (1982, 1071),
        (1983, 1072), (1984, 1073), (1985, 1074), (1986, 1075), (1987, 1076),
        (1988, 1077), (1989, 1078), (1990, 1079), (1991, 1080), (1992, 1081),
        (1993, 1082), (1994, 1047), (1995, 1048), (1996, 1049), (1997, 1050),
        (1999, 1051), (2001, 1040), (2003, 1052), (2005, 1132), (2007, 1139),
        (2009, 1152), (2011, 1156), (2013, 1164), (2015, 1183), (2017, 1187),
        (2019, 1194), (2021, 1205), (2023, 1214)]],
    *[(fid, f"Wealth {yr}") for yr, fid in [
        (1984, 1147), (1989, 1148), (1994, 1149), (1999, 1150), (2001, 1130),
        (2003, 1131), (2005, 1133), (2007, 1140)]],
    (1053, "Cross-year Individual 1968-2023"),
    (1121, "Marriage History 1985-2023"),
    (1109, "Childbirth & Adoption History 1985-2023"),
    (1123, "Parent Identification 2023"),
    (1165, "Pregnancy Intentions 2013-2023"),
    (1188, "Family Relationship Matrix 1968-2023"),
    (1175, "Active Savings 1989, 1994"),
    (1145, "Disability and Use of Time 2009"),
    (1161, "Disability and Use of Time 2013"),
    (1163, "Web Supplement 2014 Childhood Retrospective Circumstances"),
    (1177, "Web Supplement 2016 WellBeing and Daily Life"),
    (1097, "Intergenerational Transfers 1988"),
    (1159, "Intergenerational Transfers 2013"),
    (1204, "CDS Cumulative ID Map"),
    *[(fid, f"CDS {yr}") for yr, fid in [
        (1997, 1193), (2002, 1174), (2007, 1170), (2014, 1173), (2019, 1200),
        (2020, 1201), (2021, 1210), (2024, 1217)]],
    *[(fid, f"TAS {yr}") for yr, fid in [
        (2005, 1166), (2007, 1167), (2009, 1168), (2011, 1169), (2013, 1172),
        (2015, 1184), (2017, 1189), (2019, 1196), (2021, 1208), (2023, 1215)]],
]
SUPPLEMENT_PREFIXES = ("CDS", "TAS")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- http


class Session:
    def __init__(self) -> None:
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )
        self.opener.addheaders = [("User-Agent", USER_AGENT)]

    def get(self, url: str, data: dict | None = None, timeout: int = 600):
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body)
        if body is not None:
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            req.add_header("Referer", url)
        last = None
        for attempt in range(4):
            try:
                return self.opener.open(req, timeout=timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                    raise
                last = e
                wait = 2 ** (attempt + 1)
                log(f"    network error ({e}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"giving up on {url}: {last}")

    def text(self, url: str, data: dict | None = None) -> str:
        with self.get(url, data) as r:
            return r.read().decode("utf-8", errors="replace")


class FormParser(HTMLParser):
    """Collects every <input> of the page: name -> (type, value)."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: dict[str, tuple[str, str]] = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "input":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        if a.get("name"):
            self.inputs[a["name"]] = (a.get("type", "text").lower(), a.get("value", ""))


def login(session: Session, user: str, password: str) -> None:
    page = session.text(LOGIN_URL)
    form = FormParser()
    form.feed(page)
    fields = {n: v for n, (t, v) in form.inputs.items() if t == "hidden"}
    user_field = next((n for n in form.inputs if n.lower().endswith("username")), None)
    pass_field = next(
        (n for n, (t, _) in form.inputs.items() if t == "password"), None
    )
    button = next(
        (n for n, (t, _) in form.inputs.items()
         if t in ("submit", "image") and "login" in n.lower()),
        None,
    )
    if not (user_field and pass_field):
        Path("login_page.html").write_text(page)
        raise RuntimeError(
            "Could not find the username/password fields on the PSID login page; "
            "the site may have changed. Saved it as login_page.html for inspection."
        )
    fields[user_field] = user
    fields[pass_field] = password
    if button:
        fields[button] = form.inputs[button][1] or "Log In"
    result = session.text(LOGIN_URL, fields)
    if "logout" not in result.lower():
        raise RuntimeError(
            "Login failed: check your PSID username and password (the Data "
            "Center login, as on simba.isr.umich.edu/U/Login.aspx)."
        )


# --------------------------------------------------------------------------- data


class LinkParser(HTMLParser):
    """Collects (href, anchor text) pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None


def list_packaged_files(session: Session) -> list[tuple[int, str]]:
    try:
        page = session.text(ZIPMAIN_URL)
    except Exception as e:  # noqa: BLE001
        log(f"  could not read the Packaged Data page ({e}); using the built-in list")
        return FALLBACK_FILES
    parser = LinkParser()
    parser.feed(page)
    found: dict[int, str] = {}
    for href, text in parser.links:
        m = re.search(r"GetFile\w*\.aspx\?file=(\d+)", html.unescape(href or ""), re.I)
        if m:
            fid = int(m.group(1))
            found.setdefault(fid, text)
    fallback = dict(FALLBACK_FILES)
    if len(found) < 40:
        log(f"  only {len(found)} links parsed from the Packaged Data page; "
            "using the built-in list as well")
        found = {**fallback, **found}
    # Anchor text is often just a year, so prefer the descriptive built-in label.
    labels = {fid: fallback.get(fid) or text or f"file {fid}"
              for fid, text in found.items()}
    return sorted(labels.items())


def is_sensitive(label: str) -> bool:
    return "sensitive" in label.lower()


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def download_zip(session: Session, fid: int, label: str, zip_dir: Path) -> Path | None:
    existing = sorted(zip_dir.glob(f"{fid}__*.zip"))
    for p in existing:
        if zipfile.is_zipfile(p):
            log(f"  [{fid}] {label}: already have {p.name}")
            return p
        p.unlink()
    url = f"{SIMBA}/Zips/GetFileCDS.aspx?file={fid}&mainurl=Y"
    resp = session.get(url, timeout=3600)
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    fname = safe_name(m.group(1)) if m else f"{safe_name(label)}.zip"
    if not fname.lower().endswith(".zip"):
        fname += ".zip"
    tmp = zip_dir / f"{fid}__{fname}.part"
    h = hashlib.sha256()
    size = 0
    with resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            size += len(chunk)
    if not zipfile.is_zipfile(tmp):
        bad = zip_dir / f"{fid}__not_a_zip.html"
        tmp.replace(bad)
        log(f"  [{fid}] {label}: server did not return a zip (saved {bad.name}); "
            "probably logged out or the file needs an extra agreement")
        return None
    out = zip_dir / f"{fid}__{fname}"
    tmp.replace(out)
    log(f"  [{fid}] {label}: {out.name} ({size / 1e6:.1f} MB)")
    with open(zip_dir / "manifest.csv", "a", newline="") as f:
        csv.writer(f).writerow([fid, label, out.name, size, h.hexdigest(),
                                time.strftime("%Y-%m-%d %H:%M:%S")])
    return out


def step_data(args, out: Path) -> None:
    zip_dir = out / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    manifest = zip_dir / "manifest.csv"
    if not manifest.exists():
        manifest.write_text("file_id,label,filename,bytes,sha256,downloaded\n")

    session = Session()
    user = os.environ.get("PSID_USER") or input("PSID username (Data Center login): ")
    password = getpass.getpass("PSID password (not shown, not saved): ")
    log("Logging in to the PSID Data Center ...")
    login(session, user, password)
    log("  logged in")

    files = list_packaged_files(session)
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        files = [(f, l) for f, l in files if f in wanted]
    if not args.include_sensitive:
        skipped = [l for _, l in files if is_sensitive(l)]
        files = [(f, l) for f, l in files if not is_sensitive(l)]
        if skipped:
            log(f"  skipping {len(skipped)} 'Sensitive Data Variables' files")
    if args.supplements == "none":
        files = [(f, l) for f, l in files if not l.startswith(SUPPLEMENT_PREFIXES)]
    log(f"Downloading {len(files)} packaged files into {zip_dir}")

    failed = []
    for i, (fid, label) in enumerate(files, 1):
        log(f"({i}/{len(files)})")
        try:
            got = download_zip(session, fid, label, zip_dir)
            if got is None:  # one re-login, then retry once
                login(session, user, password)
                got = download_zip(session, fid, label, zip_dir)
            if got is None:
                failed.append((fid, label))
        except Exception as e:  # noqa: BLE001
            log(f"  [{fid}] {label}: FAILED ({e})")
            failed.append((fid, label))
        time.sleep(1)  # be polite to the server
    if failed:
        log("\nThese files did not download (re-run the script to retry):")
        for fid, label in failed:
            log(f"  {fid}  {label}")


# --------------------------------------------------------------------------- docs


def step_docs(args, out: Path) -> None:
    docs_dir = out / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    session = Session()
    page = session.text(DOCS_URL)
    parser = LinkParser()
    parser.feed(page)
    urls: dict[str, str] = {}
    for href, text in parser.links:
        if not href:
            continue
        url = urllib.parse.urljoin(DOCS_URL, html.unescape(href)).split("#")[0]
        host = urllib.parse.urlparse(url).netloc.lower()
        if not host.endswith("isr.umich.edu"):
            continue
        if urllib.parse.urlparse(url).path.lower().endswith(DOC_EXTENSIONS):
            urls.setdefault(url, text)
    for url in EXTRA_DOCS:
        urls.setdefault(url, "")
    if args.supplements == "none":
        urls = {u: t for u, t in urls.items()
                if not re.search(r"/(cds|tas)/", u, re.I)}
    log(f"Downloading {len(urls)} documentation files into {docs_dir}")

    index_rows = []
    failed = []
    for i, (url, text) in enumerate(sorted(urls.items()), 1):
        path = urllib.parse.unquote(urllib.parse.urlparse(url).path).lstrip("/")
        dest = docs_dir / Path(*[safe_name(p) for p in path.split("/") if p])
        index_rows.append([str(dest.relative_to(docs_dir)), text, url])
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with session.get(url) as r, open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            dest.with_suffix(dest.suffix + ".part").replace(dest)
            if i % 25 == 0:
                log(f"  {i}/{len(urls)}")
        except Exception as e:  # noqa: BLE001
            failed.append((url, str(e)))
        time.sleep(0.2)
    with open(docs_dir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["local_path", "link_text", "url"])
        w.writerows(index_rows)
    log(f"  done; index of all documents in {docs_dir / 'index.csv'}")
    if failed:
        log(f"  {len(failed)} documents failed (often dead links on the PSID page):")
        for url, err in failed:
            log(f"    {url}  ({err})")


# --------------------------------------------------------------------------- convert


def parse_sas_setup(sas_text: str):
    """Column layout, labels and character flags from a PSID SAS setup file.

    PSID setup files declare each variable in an ATTRIB block
    (``ER82001 LABEL="RELEASE NUMBER" format=f1.``; character variables have
    ``format=$char..``) and give positions in an INPUT block
    (``ER82001 1 - 1``), 1-based and inclusive. Older files use LABEL and
    ``$`` in INPUT instead; both are handled.
    Returns ([(name, start, end, is_char)], {name: label}, lrecl or None).
    """
    text = re.sub(r"/\*.*?\*/", " ", sas_text, flags=re.S)
    labels: dict[str, str] = {}
    char_vars: set[str] = set()
    for name, lab, fmt in re.findall(
        r"^\s*([A-Za-z_]\w*)\s+LABEL\s*=\s*\"([^\"]*)\"(?:\s+format\s*=\s*(\S+))?",
        text, flags=re.M | re.I,
    ):
        labels[name.upper()] = lab.strip()
        if fmt.startswith("$"):
            char_vars.add(name.upper())
    m = re.search(r"^\s*LABEL\s*$(.*?)^\s*;", text, flags=re.S | re.M | re.I)
    if m:  # old-style LABEL block: NAME = "label"
        for name, q1, q2 in re.findall(
            r"([A-Za-z_]\w*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", m.group(1)
        ):
            labels.setdefault(name.upper(), (q1 or q2).strip())
    cols = []
    m = re.search(r"^\s*INPUT\b(.*?)^\s*;", text, flags=re.S | re.M | re.I)
    if m:
        for name, dollar, start, end in re.findall(
            r"([A-Za-z_]\w*)\s+(\$?)\s*(\d+)\s*(?:-\s*(\d+))?", m.group(1)
        ):
            n = name.upper()
            s = int(start)
            cols.append((n, s, int(end) if end else s, bool(dollar) or n in char_vars))
    m = re.search(r"LRECL\s*=\s*(\d+)", text, flags=re.I)
    return cols, labels, int(m.group(1)) if m else None


def read_fixed_width(txt: Path, cols, lrecl: int | None):
    """Read a fixed-width file into a DataFrame without holding it as strings."""
    import numpy as np
    import pandas as pd

    raw = txt.read_bytes().splitlines()
    width = max(lrecl or 0, max(e for _, _, e, _ in cols), max(map(len, raw)))
    grid = np.frombuffer(b"".join(r.ljust(width) for r in raw), dtype="S1")
    grid = grid.reshape(len(raw), width)
    del raw
    out = {}
    for name, s, e, is_char in cols:
        w = e - s + 1
        field = np.ascontiguousarray(grid[:, s - 1:e]).view(f"S{w}").ravel()
        if is_char:
            out[name] = pd.array(
                [v.decode("latin-1").strip() for v in field], dtype="string"
            )
            continue
        try:
            out[name] = field.astype(np.int64)
        except ValueError:  # blanks or decimals
            out[name] = pd.to_numeric(
                pd.Series(field).str.decode("latin-1").str.strip()
                .replace("", None),
                errors="coerce",
            )
    return pd.DataFrame(out)


def convert_zip(zpath: Path, extract_root: Path, parquet_dir: Path) -> list[str]:
    stem = zpath.stem.split("__", 1)[-1]
    xdir = extract_root / stem
    if not (xdir / ".done").exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(xdir)
        (xdir / ".done").write_text("ok")
    done = []
    for sas in sorted(p for p in xdir.rglob("*") if p.suffix.lower() == ".sas"):
        name = sas.stem
        if name.lower().endswith("_formats"):
            continue
        target = parquet_dir / f"{name.upper()}.parquet"
        if target.exists():
            done.append(f"{name} (already converted)")
            continue
        txts = [p for p in xdir.rglob("*") if p.suffix.lower() == ".txt"
                and p.stem.lower() == name.lower()]
        if not txts:
            continue
        cols, labels, lrecl = parse_sas_setup(sas.read_text(errors="replace"))
        if not cols:
            log(f"    {sas.name}: no INPUT statement found, skipped")
            continue
        df = read_fixed_width(txts[0], cols, lrecl)
        df.to_parquet(target, index=False, compression="zstd")
        with open(parquet_dir / f"{name.upper()}_labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variable", "label", "start", "end", "character"])
            for n, s, e, c in cols:
                w.writerow([n, labels.get(n, ""), s, e, int(c)])
        done.append(f"{name}: {len(df):,} rows x {len(cols):,} variables")
    return done


def step_convert(args, out: Path) -> None:
    zip_dir = out / "zips"
    extract_root = out / "extracted"
    parquet_dir = out / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(p for p in zip_dir.glob("*.zip") if zipfile.is_zipfile(p))
    if not zips:
        log("No zips to convert yet; run the data step first.")
        return
    log(f"Converting {len(zips)} zips to parquet in {parquet_dir}")
    for zpath in zips:
        try:
            for line in convert_zip(zpath, extract_root, parquet_dir):
                log(f"  {line}")
        except Exception as e:  # noqa: BLE001
            log(f"  {zpath.name}: conversion FAILED ({e}); the raw files are "
                f"still in {extract_root}")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default="data,docs,convert")
    ap.add_argument("--out", default="data/raw/psid")
    ap.add_argument("--only", help="comma-separated Data Center file ids")
    ap.add_argument("--include-sensitive", action="store_true")
    ap.add_argument("--supplements", choices=["all", "none"], default="all")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps = {s.strip() for s in args.steps.split(",")}
    log(f"Output folder: {out}")
    if "data" in steps:
        step_data(args, out)
    if "docs" in steps:
        step_docs(args, out)
    if "convert" in steps:
        step_convert(args, out)
    log("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
