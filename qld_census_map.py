"""
QLD Census 2021 Postcode Map
Loads the local ABS SEIFA 2021 Excel (Postal Area level), generates synthetic
boundaries and employment data where not available, and renders an interactive
HTML choropleth for Queensland.

With full internet access to abs.gov.au the script can also download the POA
boundary shapefile and Census DataPack for real employment figures:
    python qld_census_map.py            # tries real downloads, falls back gracefully
    python qld_census_map.py --demo     # pure synthetic, skips SEIFA file too
"""

import argparse
import fnmatch
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import requests
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
SEIFA_DIR = DATA_DIR / "seifa"
BOUNDARY_DIR = DATA_DIR / "boundaries"
DATAPACK_DIR = DATA_DIR / "datapack"
OUTPUT_HTML = REPO_ROOT / "qld_census_map.html"

# Local SEIFA file uploaded by user
LOCAL_SEIFA = REPO_ROOT / "Postal Area, Indexes, SEIFA 2021.xlsx"

BOUNDARY_URL = (
    "https://www.abs.gov.au/statistics/standards/"
    "australian-statistical-geography-standard-asgs-edition-3/"
    "jul2021-jun2026/access-and-downloads/digital-boundary-files/"
    "POA_2021_AUST_SHP_GDA2020.zip"
)
DATAPACK_URL = (
    "https://www.abs.gov.au/census/find-census-data/datapacks/download/"
    "2021_GCP_POA_for_QLD_short-header.zip"
)

# Table 1 in the Postal Area SEIFA Excel has this column layout (skiprows=5):
#   POA code | IRSD score | IRSD decile | IRSAD score | IRSAD decile |
#   IER score | IER decile | IEO score | IEO decile |
#   Usual resident pop | data-caution flag | crosses-boundary flag
SEIFA_COLS = [
    "POA_CODE_2021",
    "IRSD_SCORE",   "IRSD_DECILE_AUST",
    "IRSAD_SCORE",  "IRSAD_DECILE_AUST",
    "IER_SCORE",    "IER_DECILE_AUST",
    "IEO_SCORE",    "IEO_DECILE_AUST",
    "USUAL_RESIDENT_POP",
    "_DATA_CAUTION", "_POA_CROSSES",
]

QLD_POSTCODE_RE = r"^4\d{3}$"

TOOLTIP_FIELDS = [
    "POA_CODE21", "POA_NAME21",
    "IRSD_DECILE_AUST", "IRSAD_DECILE_AUST",
    "unemployment_rate",
    "Median_tot_prsnl_inc_weekly",
    "Median_age_persons",
    "USUAL_RESIDENT_POP",
    "AREASQKM21",
]
TOOLTIP_ALIASES = [
    "Postcode:", "Area Name:",
    "IRSD Decile:", "IRSAD Decile:",
    "Unemployment %:",
    "Median Income ($/wk):",
    "Median Age:",
    "Population:",
    "Area (km²):",
]

LAYERS = [
    {
        "name": "IRSD Decile (Socio-economic Disadvantage)",
        "col": "IRSD_DECILE_AUST",
        "palette": "RdYlGn",
        "legend": "IRSD Decile (1 = Most Disadvantaged, 10 = Least)",
        "show": True,
    },
    {
        "name": "Unemployment Rate (%)",
        "col": "unemployment_rate",
        "palette": "YlOrRd",
        "legend": "Unemployment Rate (%)",
        "show": False,
    },
    {
        "name": "Median Personal Income ($/week)",
        "col": "Median_tot_prsnl_inc_weekly",
        "palette": "Blues",
        "legend": "Median Personal Income ($/week)",
        "show": False,
    },
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": "https://www.abs.gov.au/",
}

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cache] {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {dest.name} ...")
    try:
        resp = requests.get(url, stream=True, timeout=timeout, headers=_HEADERS)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} from {url}")
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc
    print(f"  [done] {dest.stat().st_size / 1e6:.1f} MB")
    return dest


# ---------------------------------------------------------------------------
# SEIFA loader  (reads the local Postal Area Excel)
# ---------------------------------------------------------------------------

def load_seifa(excel_path: Path) -> pd.DataFrame:
    """
    Read Table 1 from the ABS Postal Area SEIFA 2021 workbook.
    Data begins at row index 5 (after 3 title rows + 2 sub-header rows).
    """
    df = pd.read_excel(
        excel_path,
        sheet_name="Table 1",
        skiprows=5,
        header=None,
        names=SEIFA_COLS,
        dtype={"POA_CODE_2021": str},
    )

    # Drop footer / blank rows (non-numeric POA codes, e.g. "Source:", NaN)
    df = df.dropna(subset=["POA_CODE_2021"]).copy()
    df["POA_CODE_2021"] = df["POA_CODE_2021"].astype(str).str.strip()
    df = df[df["POA_CODE_2021"].str.match(r"^\d{4}$", na=False)]

    numeric_cols = [c for c in SEIFA_COLS if c not in ("POA_CODE_2021", "_DATA_CAUTION", "_POA_CROSSES")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DataPack extraction
# ---------------------------------------------------------------------------

def extract_datapack_csvs(zip_path: Path, dest_dir: Path) -> tuple[Path, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _find(pattern: str) -> Path | None:
        matches = list(dest_dir.rglob(pattern))
        return matches[0] if matches else None

    g17a = _find("*G17A_QLD_POA.csv")
    g02 = _find("*G02_QLD_POA.csv")
    if g17a and g02:
        print("  [cache] DataPack CSVs already extracted")
        return g17a, g02

    print("  [extract] Scanning DataPack ZIP ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        g17a_e = next((n for n in names if fnmatch.fnmatch(n, "*G17A_QLD_POA.csv")), None)
        g02_e = next((n for n in names if fnmatch.fnmatch(n, "*G02_QLD_POA.csv")), None)
        if not g17a_e or not g02_e:
            raise FileNotFoundError(f"Required CSVs not found. Sample names: {names[:10]}")
        zf.extract(g17a_e, dest_dir)
        zf.extract(g02_e, dest_dir)

    return list(dest_dir.rglob("*G17A_QLD_POA.csv"))[0], list(dest_dir.rglob("*G02_QLD_POA.csv"))[0]


def load_labour_force(g17a_path: Path) -> pd.DataFrame:
    df = pd.read_csv(g17a_path, dtype={"POA_CODE_2021": str})
    df["POA_CODE_2021"] = df["POA_CODE_2021"].str.replace(r"^POA", "", regex=True).str.strip()
    lf = next((c for c in df.columns if "LF_Tot" in c and c.startswith("P_")), None)
    un = next((c for c in df.columns if "Unemp_Tot" in c and c.startswith("P_")), None)
    if lf and un:
        df = df.rename(columns={lf: "P_LF_Tot", un: "P_Unemp_Tot"})
        df["unemployment_rate"] = np.where(
            df["P_LF_Tot"] > 0,
            (df["P_Unemp_Tot"] / df["P_LF_Tot"] * 100).round(1),
            np.nan,
        )
    else:
        df["P_LF_Tot"] = df["P_Unemp_Tot"] = df["unemployment_rate"] = np.nan
    return df[["POA_CODE_2021", "P_LF_Tot", "P_Unemp_Tot", "unemployment_rate"]]


def load_medians(g02_path: Path) -> pd.DataFrame:
    df = pd.read_csv(g02_path, dtype={"POA_CODE_2021": str})
    df["POA_CODE_2021"] = df["POA_CODE_2021"].str.replace(r"^POA", "", regex=True).str.strip()
    want = ["POA_CODE_2021", "Median_age_persons", "Median_tot_prsnl_inc_weekly",
            "Median_rent_weekly", "Median_tot_hhd_inc_weekly", "Average_household_size"]
    return df[[c for c in want if c in df.columns]]


# ---------------------------------------------------------------------------
# Boundary loader
# ---------------------------------------------------------------------------

def load_boundaries(zip_path: Path, extract_dir: Path) -> gpd.GeoDataFrame:
    shp = extract_dir / "POA_2021_AUST_GDA2020.shp"
    if not shp.exists():
        print("  [extract] Extracting boundary shapefile ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:7844")
    elif gdf.crs.to_epsg() not in (7844, 4283):
        gdf = gdf.set_crs("EPSG:7844", allow_override=True)

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf["POA_CODE21"].str.match(QLD_POSTCODE_RE, na=False)].copy()
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.001, preserve_topology=True)
    return gdf[["POA_CODE21", "POA_NAME21", "AREASQKM21", "geometry"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic geometry and employment (used when real data unavailable)
# ---------------------------------------------------------------------------

# (postcode, suburb, centre_lat, centre_lon, half_size_deg)
_QLD_POSTCODES = [
    ("4000", "Brisbane City",        -27.470, 153.025, 0.015),
    ("4005", "New Farm",             -27.463, 153.042, 0.012),
    ("4006", "Fortitude Valley",     -27.455, 153.032, 0.010),
    ("4007", "Ascot",                -27.438, 153.060, 0.013),
    ("4010", "Hamilton",             -27.440, 153.072, 0.012),
    ("4011", "Clayfield",            -27.417, 153.072, 0.014),
    ("4012", "Nundah",               -27.402, 153.073, 0.013),
    ("4017", "Brighton",             -27.300, 153.052, 0.015),
    ("4020", "Redcliffe",            -27.230, 153.102, 0.018),
    ("4051", "Alderley",             -27.423, 152.988, 0.013),
    ("4059", "Kelvin Grove",         -27.450, 152.990, 0.010),
    ("4060", "Paddington",           -27.460, 153.000, 0.012),
    ("4065", "Toowong",              -27.480, 152.982, 0.015),
    ("4067", "St Lucia",             -27.502, 153.002, 0.013),
    ("4069", "Kenmore",              -27.507, 152.940, 0.018),
    ("4101", "South Brisbane",       -27.482, 153.018, 0.010),
    ("4102", "Woolloongabba",        -27.493, 153.033, 0.012),
    ("4105", "Moorooka",             -27.534, 153.008, 0.013),
    ("4109", "Sunnybank",            -27.572, 153.051, 0.015),
    ("4113", "Eight Mile Plains",    -27.578, 153.093, 0.015),
    ("4120", "Greenslopes",          -27.495, 153.044, 0.012),
    ("4122", "Mount Gravatt",        -27.542, 153.067, 0.016),
    ("4151", "Coorparoo",            -27.493, 153.055, 0.013),
    ("4153", "Capalaba",             -27.522, 153.197, 0.018),
    ("4154", "Manly",                -27.460, 153.183, 0.016),
    ("4160", "Cleveland",            -27.526, 153.278, 0.020),
    ("4163", "Victoria Point",       -27.582, 153.299, 0.020),
    ("4170", "Cannon Hill",          -27.473, 153.103, 0.015),
    ("4114", "Logan Central",        -27.638, 153.107, 0.018),
    ("4115", "Woodridge",            -27.641, 153.132, 0.016),
    ("4127", "Springwood",           -27.608, 153.151, 0.018),
    ("4128", "Forest Lake",          -27.618, 153.003, 0.018),
    ("4205", "Beenleigh",            -27.713, 153.198, 0.018),
    ("4207", "Coomera",              -27.874, 153.298, 0.022),
    ("4209", "Ormeau",               -27.770, 153.272, 0.022),
    ("4210", "Helensvale",           -27.932, 153.352, 0.020),
    ("4211", "Nerang",               -27.980, 153.352, 0.020),
    ("4212", "Runaway Bay",          -27.903, 153.402, 0.015),
    ("4215", "Southport",            -27.960, 153.388, 0.016),
    ("4216", "Surfers Paradise",     -28.003, 153.430, 0.012),
    ("4217", "Broadbeach",           -28.030, 153.428, 0.012),
    ("4220", "Burleigh Heads",       -28.092, 153.450, 0.015),
    ("4223", "Currumbin",            -28.162, 153.491, 0.015),
    ("4224", "Coolangatta",          -28.169, 153.540, 0.015),
    ("4226", "Robina",               -28.067, 153.392, 0.020),
    ("4270", "Tamborine Mountain",   -28.034, 153.192, 0.030),
    ("4300", "Springfield",          -27.660, 152.920, 0.020),
    ("4303", "Ipswich",              -27.610, 152.772, 0.022),
    ("4305", "Bundamba",             -27.629, 152.808, 0.018),
    ("4306", "Esk",                  -27.237, 152.422, 0.060),
    ("4350", "Toowoomba",            -27.550, 151.952, 0.045),
    ("4352", "Highfields",           -27.459, 151.955, 0.040),
    ("4370", "Warwick",              -28.213, 152.037, 0.060),
    ("4390", "Goondiwindi",          -28.550, 150.310, 0.100),
    ("4400", "Dalby",                -27.183, 151.263, 0.070),
    ("4405", "Miles",                -26.659, 150.183, 0.100),
    ("4410", "Roma",                 -26.571, 148.793, 0.100),
    ("4420", "Cunnamulla",           -28.068, 145.681, 0.200),
    ("4500", "Strathpine",           -27.312, 153.042, 0.018),
    ("4502", "Kallangur",            -27.270, 153.010, 0.018),
    ("4504", "Burpengary",           -27.170, 153.003, 0.022),
    ("4505", "Caboolture",           -27.074, 153.012, 0.025),
    ("4507", "Bribie Island",        -27.040, 153.163, 0.025),
    ("4508", "Narangba",             -27.202, 153.000, 0.020),
    ("4509", "North Lakes",          -27.242, 153.020, 0.020),
    ("4510", "Woodford",             -26.958, 152.780, 0.040),
    ("4550", "Maleny",               -26.753, 152.853, 0.035),
    ("4551", "Caloundra",            -26.800, 153.130, 0.025),
    ("4552", "Beerwah",              -26.862, 153.000, 0.030),
    ("4555", "Nambour",              -26.628, 152.959, 0.025),
    ("4556", "Buderim",              -26.680, 153.062, 0.022),
    ("4557", "Maroochydore",         -26.660, 153.100, 0.020),
    ("4558", "Noosaville",           -26.397, 153.041, 0.022),
    ("4560", "Cooroy",               -26.452, 152.907, 0.030),
    ("4561", "Tewantin",             -26.402, 153.031, 0.020),
    ("4567", "Noosa Heads",          -26.391, 153.098, 0.020),
    ("4570", "Gympie",               -26.190, 152.667, 0.035),
    ("4575", "Kawana Waters",        -26.722, 153.090, 0.018),
    ("4580", "Maryborough",          -25.540, 152.700, 0.040),
    ("4655", "Hervey Bay",           -25.290, 152.840, 0.040),
    ("4670", "Bundaberg",            -24.870, 152.350, 0.045),
    ("4680", "Gladstone",            -23.840, 151.260, 0.040),
    ("4700", "Rockhampton",          -23.380, 150.510, 0.045),
    ("4701", "Emerald",              -23.524, 148.168, 0.100),
    ("4703", "Yeppoon",              -23.133, 150.743, 0.035),
    ("4720", "Longreach",            -23.440, 144.252, 0.200),
    ("4721", "Barcaldine",           -23.558, 145.290, 0.200),
    ("4730", "Charleville",          -26.402, 146.242, 0.200),
    ("4740", "Mackay",               -21.150, 149.190, 0.050),
    ("4751", "Moranbah",             -22.003, 148.050, 0.060),
    ("4800", "Bowen",                -20.012, 148.243, 0.060),
    ("4802", "Airlie Beach",         -20.292, 148.703, 0.050),
    ("4810", "Townsville",           -19.260, 146.820, 0.050),
    ("4811", "Kirwan",               -19.322, 146.751, 0.035),
    ("4814", "Hyde Park",            -19.307, 146.788, 0.030),
    ("4817", "Garbutt",              -19.284, 146.764, 0.030),
    ("4819", "Magnetic Island",      -19.139, 146.872, 0.040),
    ("4820", "Charters Towers",      -20.074, 146.263, 0.080),
    ("4825", "Mount Isa",            -20.730, 139.492, 0.150),
    ("4830", "Cloncurry",            -20.710, 140.512, 0.150),
    ("4850", "Ingham",               -18.650, 146.160, 0.060),
    ("4869", "Innisfail",            -17.523, 146.028, 0.055),
    ("4870", "Cairns",               -16.920, 145.770, 0.050),
    ("4872", "Atherton",             -17.267, 145.477, 0.060),
    ("4873", "Mossman",              -16.462, 145.372, 0.060),
    ("4874", "Cooktown",             -15.468, 145.249, 0.100),
    ("4875", "Weipa",                -12.661, 141.863, 0.150),
    ("4880", "Mareeba",              -17.001, 145.428, 0.060),
    ("4890", "Normanton",            -17.678, 141.078, 0.200),
]

_CENTROID_DICT = {poa: (lat, lon, h) for poa, _, lat, lon, h in _QLD_POSTCODES}
_NAME_DICT = {poa: name for poa, name, *_ in _QLD_POSTCODES}


def create_synthetic_boundaries(poa_codes: list[str]) -> gpd.GeoDataFrame:
    """
    Create approximate rectangular boundary polygons for QLD postcodes.
    Only postcodes in the built-in centroid lookup are returned; others
    are silently omitted (they would have highly inaccurate positions).
    """
    records = []
    for poa in poa_codes:
        if poa not in _CENTROID_DICT:
            continue
        lat, lon, half = _CENTROID_DICT[poa]
        geom = box(lon - half, lat - half * 0.7, lon + half, lat + half * 0.7)
        area_km2 = round((half * 111) ** 2 * 2, 1)
        records.append({
            "POA_CODE21": poa,
            "POA_NAME21": _NAME_DICT.get(poa, poa),
            "AREASQKM21": area_km2,
            "geometry": geom,
        })
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    print(f"  Synthetic boundaries built for {len(gdf)} postcodes")
    return gdf


def generate_synthetic_employment(df_seifa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate labour-force and medians tables correlated with real IRSD scores.
    Useful when the Census DataPack is unavailable.
    """
    rng = np.random.default_rng(42)
    poa_codes = df_seifa["POA_CODE_2021"].tolist()
    irsd = df_seifa.set_index("POA_CODE_2021")["IRSD_DECILE_AUST"]

    lf_rows, med_rows = [], []
    for poa in poa_codes:
        d = float(irsd.get(poa, 5))
        unemp = float(np.clip(rng.normal(18 - d * 1.4, 2.5), 2.0, 25.0))
        lf_tot = int(max(200, rng.normal(4000 if d >= 6 else 1500, 1000)))
        lf_rows.append({
            "POA_CODE_2021": poa,
            "P_LF_Tot": lf_tot,
            "P_Unemp_Tot": int(lf_tot * unemp / 100),
            "unemployment_rate": round(unemp, 1),
        })
        med_rows.append({
            "POA_CODE_2021": poa,
            "Median_age_persons": int(np.clip(rng.normal(36 + d * 0.5, 4), 24, 55)),
            "Median_tot_prsnl_inc_weekly": int(np.clip(rng.normal(600 + d * 70, 80), 300, 2000)),
            "Median_rent_weekly": int(np.clip(rng.normal(350 + d * 25, 50), 150, 700)),
            "Median_tot_hhd_inc_weekly": int(np.clip(rng.normal(960 + d * 110, 130), 500, 3500)),
            "Average_household_size": round(float(rng.uniform(1.8, 3.2)), 1),
        })
    return pd.DataFrame(lf_rows), pd.DataFrame(med_rows)


# ---------------------------------------------------------------------------
# Full demo mode (no local files required)
# ---------------------------------------------------------------------------

def generate_full_demo() -> gpd.GeoDataFrame:
    """Pure synthetic dataset — used when --demo is passed explicitly."""
    rng = np.random.default_rng(42)
    records = []
    for poa, suburb, lat, lon, half in _QLD_POSTCODES:
        dist = ((lat - (-27.47)) ** 2 + (lon - 153.02) ** 2) ** 0.5
        coastal = max(0.0, 1.0 - abs(lon - 153.0) / 5.0)
        base = 8.0 - dist * 1.8 + coastal * 2.0 + float(rng.normal(0, 1.2))
        irsd_d = int(np.clip(round(base), 1, 10))
        unemp = float(np.clip(rng.normal(18 - irsd_d * 1.4, 2.5), 2.0, 25.0))
        lf = int(max(200, rng.normal(4000 if irsd_d >= 6 else 1500, 1000)))
        inc = int(np.clip(rng.normal(600 + irsd_d * 70, 80), 300, 2000))
        records.append({
            "POA_CODE21": poa, "POA_NAME21": suburb,
            "AREASQKM21": round((half * 111) ** 2 * 2, 1),
            "geometry": box(lon - half, lat - half * 0.7, lon + half, lat + half * 0.7),
            "IRSD_SCORE": 900 + (irsd_d - 5) * 30,
            "IRSD_DECILE_AUST": irsd_d,
            "IRSAD_SCORE": 900 + (irsd_d - 5) * 30 + int(rng.normal(0, 10)),
            "IRSAD_DECILE_AUST": int(np.clip(irsd_d + int(rng.integers(-1, 2)), 1, 10)),
            "IER_SCORE": 900 + (irsd_d - 5) * 30 + int(rng.normal(0, 20)),
            "IER_DECILE_AUST": int(np.clip(irsd_d + int(rng.integers(-1, 2)), 1, 10)),
            "IEO_SCORE": 900 + (irsd_d - 5) * 30 + int(rng.normal(0, 20)),
            "IEO_DECILE_AUST": int(np.clip(irsd_d + int(rng.integers(-1, 2)), 1, 10)),
            "USUAL_RESIDENT_POP": int(max(200, rng.normal(8000 if irsd_d >= 6 else 2000, 3000))),
            "P_LF_Tot": lf, "P_Unemp_Tot": int(lf * unemp / 100),
            "unemployment_rate": round(unemp, 1),
            "Median_age_persons": int(np.clip(rng.normal(36 + irsd_d * 0.5, 4), 24, 55)),
            "Median_tot_prsnl_inc_weekly": inc,
            "Median_rent_weekly": int(np.clip(rng.normal(350 + irsd_d * 25, 50), 150, 700)),
            "Median_tot_hhd_inc_weekly": int(inc * 1.6),
            "Average_household_size": round(float(rng.uniform(1.8, 3.2)), 1),
        })
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    print(f"  Full demo dataset: {len(gdf)} synthetic postcodes")
    return gdf


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_datasets(
    gdf: gpd.GeoDataFrame,
    df_seifa: pd.DataFrame,
    df_lf: pd.DataFrame,
    df_med: pd.DataFrame,
) -> gpd.GeoDataFrame:
    def _left(left: gpd.GeoDataFrame, right: pd.DataFrame) -> gpd.GeoDataFrame:
        out = left.merge(right, left_on="POA_CODE21", right_on="POA_CODE_2021", how="left")
        return out.drop(columns=["POA_CODE_2021"], errors="ignore")

    gdf = _left(gdf, df_seifa)
    gdf = _left(gdf, df_lf)
    gdf = _left(gdf, df_med)

    n, ns = len(gdf), int(gdf["IRSD_DECILE_AUST"].notna().sum())
    print(f"  POA features in map: {n}")
    print(f"  SEIFA matched:       {ns}/{n}")
    if ns < n * 0.8:
        print("  WARNING: low SEIFA match rate — check POA code format")
    assert gdf.crs.to_epsg() == 4326
    return gdf


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_map(gdf: gpd.GeoDataFrame, note: str | None = None) -> folium.Map:
    m = folium.Map(
        location=[-22.0, 144.0],
        zoom_start=5,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    if note:
        folium.map.Marker(
            [-10.5, 138.5],
            icon=folium.DivIcon(
                html=(
                    f'<div style="background:rgba(255,200,0,0.9);padding:6px 10px;'
                    f'border-radius:4px;font-size:12px;white-space:nowrap;">{note}</div>'
                ),
                icon_size=(360, 36),
            ),
        ).add_to(m)

    geojson_str = gdf.to_json()

    for layer in LAYERS:
        valid = gdf[["POA_CODE21", layer["col"]]].dropna()
        cp = folium.Choropleth(
            geo_data=geojson_str,
            name=layer["name"],
            data=valid,
            columns=["POA_CODE21", layer["col"]],
            key_on="feature.properties.POA_CODE21",
            fill_color=layer["palette"],
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=layer["legend"],
            nan_fill_color="lightgrey",
            nan_fill_opacity=0.3,
            show=layer["show"],
        )
        cp.add_to(m)

        present = [f for f in TOOLTIP_FIELDS if f in gdf.columns]
        aliases = [TOOLTIP_ALIASES[TOOLTIP_FIELDS.index(f)] for f in present]
        cp.geojson.add_child(
            folium.GeoJsonTooltip(
                fields=present, aliases=aliases,
                localize=True, sticky=False, labels=True,
                style="font-size:12px;",
            )
        )

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build QLD Census 2021 postcode choropleth.")
    parser.add_argument(
        "--demo", action="store_true",
        help="Skip all files and use fully synthetic data.",
    )
    args = parser.parse_args()
    t0 = time.time()

    if args.demo:
        print("Demo mode — generating fully synthetic data ...")
        gdf_final = generate_full_demo()
        map_note = "⚠ Demo mode — fully synthetic data"
    else:
        # ---- SEIFA: use local file ----------------------------------------
        if LOCAL_SEIFA.exists():
            print(f"Loading SEIFA from local file: {LOCAL_SEIFA.name}")
            df_seifa = load_seifa(LOCAL_SEIFA)
        else:
            print(f"Local SEIFA file not found ({LOCAL_SEIFA.name}); trying download ...")
            try:
                seifa_path = download_file(
                    "https://www.abs.gov.au/statistics/people/people-and-communities/"
                    "socio-economic-indexes-areas-seifa-australia/2021/"
                    "Postal%20Area%2C%20Indexes%2C%20SEIFA%202021.xlsx",
                    SEIFA_DIR / "Postal_Area_Indexes_SEIFA_2021.xlsx",
                )
                df_seifa = load_seifa(seifa_path)
            except RuntimeError as e:
                print(f"  [!] Could not load SEIFA: {e}", file=sys.stderr)
                print("  [!] Run with --demo for fully synthetic output.", file=sys.stderr)
                sys.exit(1)

        df_seifa_qld = df_seifa[df_seifa["POA_CODE_2021"].str.match(QLD_POSTCODE_RE, na=False)].copy()
        print(f"  QLD postcodes in SEIFA: {len(df_seifa_qld)}")

        # ---- Boundaries: try download, else synthetic ----------------------
        use_synthetic_geom = False
        try:
            BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
            print("Downloading POA boundaries ...")
            boundary_zip = download_file(BOUNDARY_URL, BOUNDARY_DIR / "POA_2021_AUST_SHP_GDA2020.zip")
            extract_dir = BOUNDARY_DIR / "POA_2021_AUST_GDA2020"
            gdf = load_boundaries(boundary_zip, extract_dir)
            print(f"  Real boundary features: {len(gdf)}")
        except RuntimeError:
            print("  [!] Boundary download unavailable — using approximate rectangular polygons.")
            use_synthetic_geom = True
            gdf = create_synthetic_boundaries(df_seifa_qld["POA_CODE_2021"].tolist())

        # ---- Employment: try download, else synthetic ----------------------
        use_synthetic_emp = False
        try:
            DATAPACK_DIR.mkdir(parents=True, exist_ok=True)
            print("Downloading Census DataPack (QLD POA) ...")
            datapack_zip = download_file(DATAPACK_URL, DATAPACK_DIR / "2021_GCP_POA_for_QLD_short-header.zip")
            g17a, g02 = extract_datapack_csvs(datapack_zip, DATAPACK_DIR)
            df_lf = load_labour_force(g17a)
            df_med = load_medians(g02)
            print(f"  Real DataPack rows: {len(df_lf)}")
        except RuntimeError:
            print("  [!] DataPack download unavailable — generating synthetic employment data.")
            use_synthetic_emp = True
            df_lf, df_med = generate_synthetic_employment(df_seifa_qld)

        print("Merging datasets ...")
        gdf_final = merge_datasets(gdf, df_seifa_qld, df_lf, df_med)

        parts = []
        if use_synthetic_geom:
            parts.append("approximate boundaries")
        if use_synthetic_emp:
            parts.append("synthetic employment")
        map_note = ("ℹ Real SEIFA 2021 data — " + ", ".join(parts)) if parts else None

    print("Building map ...")
    m = build_map(gdf_final, note=map_note)
    m.save(str(OUTPUT_HTML))

    elapsed = time.time() - t0
    print(f"\nDone!  {OUTPUT_HTML}")
    print(f"Size:  {OUTPUT_HTML.stat().st_size / 1e6:.1f} MB")
    print(f"Time:  {elapsed:.1f}s")
    if map_note:
        print(f"Note:  {map_note}")


if __name__ == "__main__":
    main()
