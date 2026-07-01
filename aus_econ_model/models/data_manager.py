"""
Data Manager — pull, cache, and serve Australian economic time series.

Sources:
    RBA Statistical Tables (CSV direct links)
    ABS Data API (SDMX-JSON 2.0 + SDMX-XML for structure)
    RBA Chart Pack

Caching:
    All data cached as Parquet in data/rba/ and data/abs/.
    Cache refreshed when > cache_max_age_days old (default 1).
"""

import csv
import io
import json
import re
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable, Union
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests


# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RBA_DIR = DATA_DIR / "rba"
ABS_DIR = DATA_DIR / "abs"
MANIFEST_PATH = DATA_DIR / "MANIFEST.md"


# ── Source Registry ──────────────────────────────────────────────────────────

RBA_SOURCES = {
    "A1": {
        "url": "https://www.rba.gov.au/statistics/tables/csv/a1-data.csv",
        "name": "Reserve Bank of Australia - Balance Sheet",
        "frequency": "Weekly",
        "notes": "RBA balance sheet items ($m). Series: Notes on issue, ES balances, Govt deposits, etc.",
    },
    "D2": {
        "url": "https://www.rba.gov.au/statistics/tables/csv/d2-data.csv",
        "name": "Lending and Credit Aggregates",
        "frequency": "Monthly",
        "notes": "Private debt proxy. Columns: Total credit, Housing credit, Business credit, Personal credit, etc.",
    },
    "E1": {
        "url": "https://www.rba.gov.au/statistics/tables/csv/e1-data.csv",
        "name": "Household and Business Balance Sheets",
        "frequency": "Quarterly",
        "notes": "Household assets, liabilities, net worth. Includes housing, deposits, loans, superannuation.",
    },
    "G1": {
        "url": "https://www.rba.gov.au/statistics/tables/csv/g1-data.csv",
        "name": "Consumer Price Inflation",
        "frequency": "Quarterly",
        "notes": "CPI measures (ABS sourced). Headline and underlying inflation.",
    },
    "G3": {
        "url": "https://www.rba.gov.au/statistics/tables/csv/g3-data.csv",
        "name": "Wage and Labour Statistics",
        "frequency": "Quarterly/Monthly",
        "notes": "WPI, labour force, unemployment rate, participation rate, etc.",
    },
}

ABS_SOURCES = {
    "CPI": {
        "dataflow": "CPI",
        "name": "Consumer Price Index (6401.0)",
        "frequency": "Quarterly",
        "notes": "CPI index for all groups. Dimensions: MEASURE, INDEX, TSEST, REGION, FREQ.",
    },
    "LF": {
        "dataflow": "LF",
        "name": "Labour Force (6202.0)",
        "frequency": "Monthly",
        "notes": "Employment, unemployment, participation rate by age and sex.",
    },
    "WPI": {
        "dataflow": "WPI",
        "name": "Wage Price Index (6345.0)",
        "frequency": "Quarterly",
        "notes": "Wage price index by sector and industry (wage growth indicator).",
    },
    "LCI": {
        "dataflow": "LCI",
        "name": "Labour Costs Index (6345.0)",
        "frequency": "Quarterly",
        "notes": "Labour cost index including wages, salaries, and non-wage costs.",
    },
    "AWE": {
        "dataflow": "AWE",
        "name": "Average Weekly Earnings (6302.0)",
        "frequency": "Bi-annual",
        "notes": "Average weekly earnings by sector and sex.",
    },
}


# ── Date Parsing ─────────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%d/%m/%Y",       # 30/09/1976
    "%d-%b-%Y",       # 03-Jul-2013
    "%Y-%m-%d",       # 2013-07-03
    "%Y-%m",          # 2013-07
    "%Y%m",           # 201307
    "%b-%Y",          # Jul-2013
    "%Y-Q%q",         # 2024-Q3
    "%Y",             # 2024
]


def parse_date(val) -> pd.Timestamp:
    """Parse various date formats used by RBA/ABS. Returns NaT on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NaT
    val = str(val).strip().strip('"').strip("'")
    if not val or val in ("...", "NA", "N/A", "-", ""):
        return pd.NaT
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(val, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        result = pd.to_datetime(val, infer_datetime_format=True)
        if pd.notna(result):
            return result
    except Exception:
        pass
    m = re.match(r"(\d{4})", val)
    if m:
        try:
            return pd.to_datetime(m.group(1), format="%Y")
        except Exception:
            pass
    return pd.NaT


# ── RBA CSV Parser ───────────────────────────────────────────────────────────


def _find_header_row(lines: list[str]):
    """Find the Series ID row in an RBA CSV."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Series ID") or stripped.startswith("Series_ID"):
            return i, i + 1
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        first = line.split(",")[0].strip('"').strip("'")
        if first and (re.match(r"\d{2}[/-]", first) or re.match(r"\d{4}[-/]", first)):
            return max(0, i - 1), i
    return 10, 11


def _parse_rba_csv(content: str) -> pd.DataFrame:
    """Parse an RBA statistical CSV into a tidy DataFrame."""
    lines = content.splitlines()
    header_idx, data_start = _find_header_row(lines)
    if header_idx < 0 or header_idx >= len(lines):
        return pd.DataFrame()
    header_line = lines[header_idx]
    try:
        headers = next(csv.reader([header_line]))
    except Exception:
        headers = header_line.split(",")
    headers = [h.strip().replace(" ", "_").replace("/", "_per_") for h in headers]
    data_lines = lines[data_start:]
    while data_lines and not data_lines[-1].strip():
        data_lines.pop()
    if not data_lines:
        return pd.DataFrame()
    raw = io.StringIO("\n".join(data_lines))
    df = pd.read_csv(raw, header=None, names=headers, low_memory=False,
                     skipinitialspace=True, quoting=csv.QUOTE_ALL)
    old_name = df.columns[0]
    df = df.rename(columns={old_name: "date"})
    df["date"] = df["date"].apply(parse_date)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # ensure datetime64 type, not object
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── ABS CSV Parser ──────────────────────────────────────────────────────────

_FREQ_TO_MONTHS = {
    "A": 12, "S": 6, "Q": 3, "M": 1, "W": None, "D": None,
}


def _parse_abs_period(period: str) -> pd.Timestamp:
    """Parse ABS time period formats used in CSV output.

    Examples: '2018-09', '1997-Q3', '1945', '2018-09-30', '2020-01-01'
    """
    if pd.isna(period) or period is None:
        return pd.NaT
    p = str(period).strip()
    if not p:
        return pd.NaT

    # Try YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", p):
        try:
            return pd.to_datetime(p, format="%Y-%m-%d")
        except Exception:
            pass

    # Try YYYY-MM (monthly)
    if re.match(r"^\d{4}-\d{2}$", p):
        try:
            return pd.to_datetime(p + "-01", format="%Y-%m-%d")
        except Exception:
            pass

    # Try YYYY-QN (quarterly)
    m = re.match(r"^(\d{4})-Q([1-4])$", p)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        month = (q - 1) * 3 + 1
        return pd.Timestamp(year=year, month=month, day=1)

    # Try YYYY-SN (half-yearly)
    m = re.match(r"^(\d{4})-S([12])$", p)
    if m:
        year, s = int(m.group(1)), int(m.group(2))
        month = (s - 1) * 6 + 1
        return pd.Timestamp(year=year, month=month, day=1)

    # Try YYYY (annual)
    if re.match(r"^\d{4}$", p):
        return pd.Timestamp(year=int(p), month=1, day=1)

    # Fallback
    return parse_date(p)


def _parse_abs_csv(content: str) -> pd.DataFrame:
    """Parse an ABS SDMX CSV response into a tidy DataFrame.

    The ABS CSV format has dimension columns plus TIME_PERIOD and OBS_VALUE.
    """
    lines = content.splitlines()
    if not lines:
        return pd.DataFrame()

    # Simple CSV parse using pandas
    raw = io.StringIO(content)
    try:
        df = pd.read_csv(raw, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # Rename key columns
    rename = {}
    if "OBS_VALUE" in df.columns:
        rename["OBS_VALUE"] = "value"
    if "TIME_PERIOD" in df.columns:
        rename["TIME_PERIOD"] = "period"

    if rename:
        df = df.rename(columns=rename)

    # Drop non-data columns that we don't need
    drop_cols = ["DATAFLOW", "UNIT_MEASURE", "UNIT_MULT", "OBS_STATUS",
                 "OBS_COMMENT", "DECIMALS", "BASE_PERIOD", "OBS_CONF"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Parse period column as dates
    if "period" in df.columns:
        df["date"] = df["period"].apply(_parse_abs_period)
        # Separate out rows with bad dates
        valid_dates = df["date"].notna()
        if valid_dates.sum() == 0:
            df = df.drop(columns=["date"])
        else:
            df = df.drop(columns=["period"])
            # Drop rows with unparseable dates
            df = df[valid_dates].copy()

    # Convert value column to numeric
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Remove rows with missing values
    if "value" in df.columns:
        df = df.dropna(subset=["value"])

    return df.reset_index(drop=True)


def _fetch_and_parse_abs(
    dataflow: str,
    params: Optional[dict] = None,
    timeout: int = 120,
) -> pd.DataFrame:
    """Fetch ABS SDMX data as CSV and parse into a tidy DataFrame.

    Uses CSV format which includes explicit time periods and dimension values.
    """
    query_params = {"format": "csv"}
    if params:
        query_params.update(params)

    url = f"https://api.data.abs.gov.au/data/{dataflow}"
    resp = requests.get(url, params=query_params, timeout=timeout)
    resp.raise_for_status()

    return _parse_abs_csv(resp.text)


# ── The Main Event ────────────────────────────────────────────────────────────


class DataManager:
    """Download, cache, and serve Australian economic time series.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Override the default cache directory.
    cache_max_age_days : int, default 1
        Days before cached Parquet files are refreshed.
    """

    def __init__(self, cache_dir: Optional[Path] = None, cache_max_age_days: int = 1):
        self.data_dir = Path(cache_dir) if cache_dir else DATA_DIR
        self.rba_dir = self.data_dir / "rba"
        self.abs_dir = self.data_dir / "abs"
        self.cache_max_age = timedelta(days=cache_max_age_days)

        self.rba_dir.mkdir(parents=True, exist_ok=True)
        self.abs_dir.mkdir(parents=True, exist_ok=True)

        # In-memory session cache
        self._cache: dict[str, pd.DataFrame] = {}

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _cache_path(self, source: str, name: str) -> Path:
        if source == "rba":
            return self.rba_dir / f"{name}.parquet"
        elif source == "abs":
            return self.abs_dir / f"{name}.parquet"
        sub = self.data_dir / source
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{name}.parquet"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < self.cache_max_age

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _write_cache(self, path: Path, df: pd.DataFrame):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Normalise types to avoid pyarrow conversion failures on mixed object columns
        df2 = df.copy()
        for col in df2.select_dtypes(include=["object"]).columns:
            # Attempt to coerce datetime-like object columns to datetime64
            try:
                coerced = pd.to_datetime(df2[col], errors="coerce")
                if coerced.notna().sum() > len(df2) * 0.5:
                    df2[col] = coerced
            except Exception:
                pass
        df2.to_parquet(path, index=False)

    # ── Core downloader ────────────────────────────────────────────────────

    def download_and_cache(
        self,
        url: str,
        name: str,
        source: str = "rba",
        force: bool = False,
        parser: Optional[Callable] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Download data from *url*, cache as Parquet, return a DataFrame.

        Parameters
        ----------
        url : str
            URL to download.
        name : str
            Cache name (e.g. 'D2', 'CPI').
        source : str
            Subdirectory: 'rba', 'abs', or custom.
        force : bool
            Re-download even if cache is fresh.
        parser : callable, optional
            Custom parser(response) -> DataFrame. Auto-detected if None.
        **kwargs
            Extra args passed to requests.get().

        Returns
        -------
        pd.DataFrame
        """
        cache_key = f"{source}:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cache_path = self._cache_path(source, name)
        if not force and self._is_fresh(cache_path):
            cached = self._read_cache(cache_path)
            if cached is not None:
                self._cache[cache_key] = cached
                return cached

        try:
            resp = requests.get(url, timeout=kwargs.pop("timeout", 120), **kwargs)
            resp.raise_for_status()
        except requests.RequestException as e:
            cached = self._read_cache(cache_path)
            if cached is not None:
                print(f"Download failed ({e}), using stale cache for {source}:{name}")
                return cached
            raise RuntimeError(f"Failed to download {url}: {e}")

        if parser is not None:
            df = parser(resp)
        elif source == "rba":
            df = _parse_rba_csv(resp.text)
        elif source == "abs":
            df = _parse_abs_csv(resp.text)
        else:
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

        if df.empty:
            raise RuntimeError(f"Downloaded {url} but parsed into empty DataFrame")

        self._write_cache(cache_path, df)
        self._cache[cache_key] = df
        return df

    # ── RBA helpers ────────────────────────────────────────────────────────

    def fetch_rba_table(self, code: str, force: bool = False) -> pd.DataFrame:
        """Fetch an RBA statistical table by code (e.g. 'D2', 'A1', 'E1')."""
        if code not in RBA_SOURCES:
            raise ValueError(f"Unknown RBA table: {code}. Known: {list(RBA_SOURCES.keys())}")
        info = RBA_SOURCES[code]
        return self.download_and_cache(
            url=info["url"], name=code, source="rba", force=force,
        )

    # ── ABS helpers ────────────────────────────────────────────────────────

    def fetch_abs_dataflow(
        self,
        dataflow: str,
        force: bool = False,
        timeout: int = 120,
    ) -> pd.DataFrame:
        """Fetch data from an ABS SDMX dataflow with caching.

        Parameters
        ----------
        dataflow : str
            Dataflow ID (e.g. 'CPI', 'LF', 'WPI').
        force : bool
            Re-download even if cache is fresh.
        timeout : int
            HTTP timeout in seconds.

        Returns
        -------
        pd.DataFrame
        """
        cache_key = f"abs:{dataflow}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cache_path = self._cache_path("abs", dataflow)
        if not force and self._is_fresh(cache_path):
            cached = self._read_cache(cache_path)
            if cached is not None:
                self._cache[cache_key] = cached
                return cached

        try:
            df = _fetch_and_parse_abs(dataflow, timeout=timeout)
        except Exception as e:
            cached = self._read_cache(cache_path)
            if cached is not None:
                print(f"ABS {dataflow} fetch failed ({e}), using stale cache")
                return cached
            raise RuntimeError(f"Failed to fetch ABS {dataflow}: {e}")

        if df.empty:
            raise RuntimeError(f"ABS {dataflow} returned empty data")

        self._write_cache(cache_path, df)
        self._cache[cache_key] = df
        return df

    # ── Indicator Shortcuts ─────────────────────────────────────────────────

    def get_gdp(self, force: bool = False) -> pd.DataFrame:
        """Get GDP-related data.

        Primary: RBA Table A1 (balance sheet items → economic activity proxy).
        Returns a DataFrame with date and various RBA balance sheet columns.
        """
        return self.fetch_rba_table("A1", force=force)

    def get_private_debt_gdp(self, force: bool = False) -> pd.DataFrame:
        """Get private debt aggregates from RBA D2.

        D2 contains lending and credit aggregates: total credit,
        housing credit, business credit, personal credit ($billion).
        """
        return self.fetch_rba_table("D2", force=force)

    def get_household_debt(self, force: bool = False) -> pd.DataFrame:
        """Get household balance sheet data from RBA E1.

        E1 includes household assets (dwellings, deposits, superannuation)
        and liabilities (housing loans, other loans, total debt).
        """
        return self.fetch_rba_table("E1", force=force)

    def get_cpi(self, force: bool = False) -> pd.DataFrame:
        """Get CPI data.

        Primary: ABS CPI (6401.0) via SDMX.
        Fallback: RBA G1 table (ABS-sourced CPI measures).
        """
        try:
            return self.fetch_abs_dataflow("CPI", force=force)
        except Exception as e:
            print(f"ABS CPI unavailable ({e}), falling back to RBA G1")
            return self.fetch_rba_table("G1", force=force)

    def get_wage_share(self, force: bool = False) -> pd.DataFrame:
        """Get wage/labour cost data as a proxy for wage share.

        Wage share = compensation of employees / GDP requires National
        Accounts data (5206.0), which is not available via the ABS API.

        Returns WPI (Wage Price Index) data from ABS or RBA G3 as proxy.
        """
        try:
            return self.fetch_abs_dataflow("WPI", force=force)
        except Exception as e:
            print(f"ABS WPI unavailable ({e}), falling back to RBA G3")
            return self.fetch_rba_table("G3", force=force)

    def get_employment_rate(self, force: bool = False) -> pd.DataFrame:
        """Get employment/labour force data.

        Primary: ABS LF (Labour Force 6202.0).
        Fallback: RBA G3 table (labour market indicators).
        """
        try:
            return self.fetch_abs_dataflow("LF", force=force)
        except Exception as e:
            print(f"ABS LF unavailable ({e}), falling back to RBA G3")
            return self.fetch_rba_table("G3", force=force)

    # ── Bulk download ──────────────────────────────────────────────────────

    def download_all(self, force: bool = False) -> dict[str, pd.DataFrame]:
        """Download all known RBA tables and key ABS dataflows.

        Returns dict mapping source name -> DataFrame.
        """
        results = {}
        for code in RBA_SOURCES:
            try:
                results[f"rba_{code}"] = self.fetch_rba_table(code, force=force)
                print(f"  ✓ RBA {code}")
            except Exception as e:
                print(f"  ✗ RBA {code}: {e}")
        for name in ABS_SOURCES:
            try:
                results[f"abs_{name}"] = self.fetch_abs_dataflow(name, force=force)
                print(f"  ✓ ABS {name}")
            except Exception as e:
                print(f"  ✗ ABS {name}: {e}")
        return results

    # ── Async background download ──────────────────────────────────────────

    def download_all_async(
        self,
        force: bool = False,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> threading.Thread:
        """Download all data in a background thread.

        Parameters
        ----------
        force : bool
            Force re-download.
        callback : callable, optional
            Called with results dict when done.

        Returns
        -------
        threading.Thread (already started).
        """
        results: dict = {}
        jobs = []
        for code in RBA_SOURCES:
            jobs.append(("rba_" + code, lambda c=code: self.fetch_rba_table(c, force=force)))
        for name in ABS_SOURCES:
            jobs.append(("abs_" + name, lambda n=name: self.fetch_abs_dataflow(n, force=force)))

        def _bg_worker():
            for jname, method in jobs:
                try:
                    results[jname] = method()
                except Exception as e:
                    results[jname] = e
            if callback:
                callback(results)

        t = threading.Thread(target=_bg_worker, daemon=True)
        t.start()
        return t

    # ── Summary ────────────────────────────────────────────────────────────

    def data_summary(self) -> dict:
        """Return a dict summarising what data is cached/available.

        Each entry: {
            'source': str, 'name': str,
            'rows': int, 'cols': list, 'date_min': str, 'date_max': str,
            'frequency': str
        }
        """
        summary = {}
        for code, info in RBA_SOURCES.items():
            cache_path = self._cache_path("rba", code)
            df = self._read_cache(cache_path)
            entry = {"source": f"RBA {code}", "name": info["name"], "frequency": info["frequency"]}
            if df is not None and not df.empty:
                entry["rows"] = len(df)
                entry["cols"] = list(df.columns[:8])
                entry["date_min"] = str(df["date"].min().date()) if "date" in df.columns and df["date"].notna().any() else "—"
                entry["date_max"] = str(df["date"].max().date()) if "date" in df.columns and df["date"].notna().any() else "—"
            else:
                entry["status"] = "not cached"
            summary[f"rba_{code}"] = entry

        for name, info in ABS_SOURCES.items():
            cache_path = self._cache_path("abs", name)
            df = self._read_cache(cache_path)
            entry = {"source": f"ABS {info['dataflow']}", "name": info["name"], "frequency": info["frequency"]}
            if df is not None and not df.empty:
                entry["rows"] = len(df)
                entry["cols"] = list(df.columns[:8])
                if "date" in df.columns and df["date"].notna().any():
                    entry["date_min"] = str(df["date"].min().date())
                    entry["date_max"] = str(df["date"].max().date())
                else:
                    entry["date_min"] = "—"
                    entry["date_max"] = "—"
            else:
                entry["status"] = "not cached"
            summary[f"abs_{name}"] = entry

        return summary

    def print_summary(self):
        """Print a human-readable summary of available data."""
        summary = self.data_summary()
        print("=" * 72)
        print("  Australian Economic Data — Available Sources")
        print("=" * 72)
        for key, info in sorted(summary.items()):
            if info.get("status") == "not cached":
                print(f"  [{info['source']:>8}] {info['name']}")
                print(f"           ⚠  Not cached — run download_all()")
            else:
                dr = f"{info.get('date_min','—')} → {info.get('date_max','—')}"
                print(f"  [{info['source']:>8}] {info['name']}")
                print(f"           {info.get('rows',0):>5} rows  |  {info.get('frequency','?'):>12}  |  {dr}")
        print("=" * 72)


# ── Module-level convenience ────────────────────────────────────────────────

_default_manager = None


def get_manager(cache_dir: Optional[Path] = None, cache_max_age_days: int = 1) -> DataManager:
    """Get or create the default DataManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DataManager(cache_dir=cache_dir, cache_max_age_days=cache_max_age_days)
    return _default_manager


# ── Standalone ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dm = DataManager()
    print("Downloading all available data...")
    results = dm.download_all(force=False)
    print(f"\nFetched {len(results)} / {len(RBA_SOURCES) + len(ABS_SOURCES)} datasets")
    dm.print_summary()
