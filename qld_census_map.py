"""
QLD Census 2021 Postcode Map
Downloads SEIFA 2021 + employment DataPack + POA boundaries from ABS,
merges them, and produces an interactive HTML choropleth for Queensland.

In environments without internet access, pass --demo to generate a map
from built-in synthetic data matching the same schema.
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
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
SEIFA_DIR = DATA_DIR / "seifa"
BOUNDARY_DIR = DATA_DIR / "boundaries"
DATAPACK_DIR = DATA_DIR / "datapack"
OUTPUT_HTML = Path(__file__).parent / "qld_census_map.html"

SEIFA_URL = (
    "https://www.abs.gov.au/statistics/people/people-and-communities/"
    "socio-economic-indexes-areas-seifa-australia/2021/"
    "Postal%20Area%2C%20Indexes%2C%20SEIFA%202021.xlsx"
)
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

SEIFA_COLS = [
    "POA_CODE_2021", "POA_NAME_2021",
    "IRSD_SCORE", "IRSD_DECILE_AUST", "IRSD_RANK_AUST",
    "IRSAD_SCORE", "IRSAD_DECILE_AUST", "IRSAD_RANK_AUST",
    "IEO_SCORE", "IEO_DECILE_AUST", "IEO_RANK_AUST",
    "IER_SCORE", "IER_DECILE_AUST", "IER_RANK_AUST",
    "USUAL_RESIDENT_POP",
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
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc
    print(f"  [done] {dest.stat().st_size / 1e6:.1f} MB")
    return dest


# ---------------------------------------------------------------------------
# SEIFA loader
# ---------------------------------------------------------------------------

def load_seifa(excel_path: Path) -> pd.DataFrame:
    def _read(skiprows: int) -> pd.DataFrame:
        return pd.read_excel(
            excel_path,
            sheet_name="Table 1",
            skiprows=skiprows,
            header=None,
            names=SEIFA_COLS,
            dtype={"POA_CODE_2021": str},
        )

    df = _read(6)
    if len(df.columns) != len(SEIFA_COLS):
        df = _read(5)

    df = df.dropna(subset=["POA_CODE_2021"]).copy()
    df["POA_CODE_2021"] = (
        df["POA_CODE_2021"].str.replace(r"^POA", "", regex=True).str.strip()
    )
    for col in SEIFA_COLS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["POA_CODE_2021"].str.match(r"^\d{4}$", na=False)]
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
        g17a_entry = next((n for n in names if fnmatch.fnmatch(n, "*G17A_QLD_POA.csv")), None)
        g02_entry = next((n for n in names if fnmatch.fnmatch(n, "*G02_QLD_POA.csv")), None)
        if not g17a_entry:
            raise FileNotFoundError(f"G17A_QLD_POA.csv not in ZIP. Sample entries: {names[:10]}")
        if not g02_entry:
            raise FileNotFoundError(f"G02_QLD_POA.csv not in ZIP. Sample entries: {names[:10]}")
        zf.extract(g17a_entry, dest_dir)
        zf.extract(g02_entry, dest_dir)

    return list(dest_dir.rglob("*G17A_QLD_POA.csv"))[0], list(dest_dir.rglob("*G02_QLD_POA.csv"))[0]


# ---------------------------------------------------------------------------
# Tabular loaders
# ---------------------------------------------------------------------------

def load_labour_force(g17a_path: Path) -> pd.DataFrame:
    df = pd.read_csv(g17a_path, dtype={"POA_CODE_2021": str})
    df["POA_CODE_2021"] = df["POA_CODE_2021"].str.replace(r"^POA", "", regex=True).str.strip()

    lf_col = next((c for c in df.columns if "LF_Tot" in c and c.startswith("P_")), None)
    unemp_col = next((c for c in df.columns if "Unemp_Tot" in c and c.startswith("P_")), None)
    if lf_col and unemp_col:
        df = df.rename(columns={lf_col: "P_LF_Tot", unemp_col: "P_Unemp_Tot"})
        df["unemployment_rate"] = np.where(
            df["P_LF_Tot"] > 0,
            (df["P_Unemp_Tot"] / df["P_LF_Tot"] * 100).round(1),
            np.nan,
        )
    else:
        df["P_LF_Tot"] = np.nan
        df["P_Unemp_Tot"] = np.nan
        df["unemployment_rate"] = np.nan

    keep = [c for c in ["POA_CODE_2021", "P_LF_Tot", "P_Unemp_Tot", "unemployment_rate"] if c in df.columns]
    return df[keep]


def load_medians(g02_path: Path) -> pd.DataFrame:
    df = pd.read_csv(g02_path, dtype={"POA_CODE_2021": str})
    df["POA_CODE_2021"] = df["POA_CODE_2021"].str.replace(r"^POA", "", regex=True).str.strip()
    want = [
        "POA_CODE_2021", "Median_age_persons", "Median_tot_prsnl_inc_weekly",
        "Median_rent_weekly", "Median_tot_hhd_inc_weekly", "Average_household_size",
    ]
    return df[[c for c in want if c in df.columns]]


# ---------------------------------------------------------------------------
# Boundary loader
# ---------------------------------------------------------------------------

def load_boundaries(zip_path: Path, extract_dir: Path) -> gpd.GeoDataFrame:
    shp_path = extract_dir / "POA_2021_AUST_GDA2020.shp"
    if not shp_path.exists():
        print("  [extract] Extracting boundary shapefile ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:7844")
    elif gdf.crs.to_epsg() not in (7844, 4283):
        gdf = gdf.set_crs("EPSG:7844", allow_override=True)

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf["POA_CODE21"].str.match(QLD_POSTCODE_RE, na=False)].copy()
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.001, preserve_topology=True)
    return gdf[["POA_CODE21", "POA_NAME21", "AREASQKM21", "geometry"]].reset_index(drop=True)


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

    n, ns = len(gdf), gdf["IRSD_DECILE_AUST"].notna().sum()
    print(f"  QLD POA features:  {n}")
    print(f"  SEIFA matched:     {ns}/{n}")
    if ns < n * 0.8:
        print(f"  WARNING: low SEIFA match rate")
    assert 300 <= n <= 700, f"Unexpected POA count: {n}"
    assert gdf.crs.to_epsg() == 4326
    return gdf


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

# (postcode, suburb_name, centre_lat, centre_lon, half_size_deg)
# half_size determines the rectangular polygon extent — smaller for dense urban,
# larger for sparse rural areas.
_QLD_POSTCODES = [
    # Brisbane inner
    ("4000", "Brisbane City",        -27.470, 153.025, 0.015),
    ("4005", "New Farm",             -27.463, 153.042, 0.012),
    ("4006", "Fortitude Valley",     -27.455, 153.032, 0.010),
    ("4007", "Ascot",                -27.438, 153.060, 0.013),
    ("4010", "Hamilton",             -27.440, 153.072, 0.012),
    ("4011", "Clayfield",            -27.417, 153.072, 0.014),
    ("4012", "Nundah",               -27.402, 153.073, 0.013),
    ("4017", "Brighton",             -27.300, 153.052, 0.015),
    ("4020", "Redcliffe",            -27.230, 153.102, 0.018),
    # Brisbane west/inner
    ("4051", "Alderley",             -27.423, 152.988, 0.013),
    ("4059", "Kelvin Grove",         -27.450, 152.990, 0.010),
    ("4060", "Paddington",           -27.460, 153.000, 0.012),
    ("4065", "Toowong",              -27.480, 152.982, 0.015),
    ("4067", "St Lucia",             -27.502, 153.002, 0.013),
    ("4069", "Kenmore",              -27.507, 152.940, 0.018),
    # Brisbane south
    ("4101", "South Brisbane",       -27.482, 153.018, 0.010),
    ("4102", "Woolloongabba",        -27.493, 153.033, 0.012),
    ("4105", "Moorooka",             -27.534, 153.008, 0.013),
    ("4109", "Sunnybank",            -27.572, 153.051, 0.015),
    ("4113", "Eight Mile Plains",    -27.578, 153.093, 0.015),
    ("4120", "Greenslopes",          -27.495, 153.044, 0.012),
    ("4122", "Mount Gravatt",        -27.542, 153.067, 0.016),
    ("4151", "Coorparoo",            -27.493, 153.055, 0.013),
    # Brisbane east
    ("4153", "Capalaba",             -27.522, 153.197, 0.018),
    ("4154", "Manly",                -27.460, 153.183, 0.016),
    ("4160", "Cleveland",            -27.526, 153.278, 0.020),
    ("4163", "Victoria Point",       -27.582, 153.299, 0.020),
    ("4170", "Cannon Hill",          -27.473, 153.103, 0.015),
    # Logan / South of Brisbane
    ("4114", "Logan Central",        -27.638, 153.107, 0.018),
    ("4115", "Woodridge",            -27.641, 153.132, 0.016),
    ("4127", "Springwood",           -27.608, 153.151, 0.018),
    ("4205", "Beenleigh",            -27.713, 153.198, 0.018),
    # Gold Coast
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
    # Ipswich
    ("4300", "Springfield",          -27.660, 152.920, 0.020),
    ("4303", "Ipswich",              -27.610, 152.772, 0.022),
    ("4305", "Bundamba",             -27.629, 152.808, 0.018),
    # Moreton Bay / North of Brisbane
    ("4500", "Strathpine",           -27.312, 153.042, 0.018),
    ("4502", "Kallangur",            -27.270, 153.010, 0.018),
    ("4504", "Burpengary",           -27.170, 153.003, 0.022),
    ("4505", "Caboolture",           -27.074, 153.012, 0.025),
    ("4508", "Narangba",             -27.202, 153.000, 0.020),
    ("4509", "North Lakes",          -27.242, 153.020, 0.020),
    # Sunshine Coast
    ("4551", "Caloundra",            -26.800, 153.130, 0.025),
    ("4555", "Nambour",              -26.628, 152.959, 0.025),
    ("4556", "Buderim",              -26.680, 153.062, 0.022),
    ("4557", "Maroochydore",         -26.660, 153.100, 0.020),
    ("4558", "Noosaville",           -26.397, 153.041, 0.022),
    ("4567", "Noosa Heads",          -26.391, 153.098, 0.020),
    ("4570", "Gympie",               -26.190, 152.667, 0.035),
    ("4575", "Kawana Waters",        -26.722, 153.090, 0.018),
    # Fraser Coast
    ("4580", "Maryborough",          -25.540, 152.700, 0.040),
    ("4655", "Hervey Bay",           -25.290, 152.840, 0.040),
    # Bundaberg / Wide Bay
    ("4670", "Bundaberg",            -24.870, 152.350, 0.045),
    # Gladstone
    ("4680", "Gladstone",            -23.840, 151.260, 0.040),
    # Rockhampton
    ("4700", "Rockhampton",          -23.380, 150.510, 0.045),
    ("4703", "Yeppoon",              -23.133, 150.743, 0.035),
    # Mackay
    ("4740", "Mackay",               -21.150, 149.190, 0.050),
    ("4751", "Moranbah",             -22.003, 148.050, 0.060),
    # Whitsunday
    ("4800", "Bowen",                -20.012, 148.243, 0.060),
    ("4802", "Airlie Beach",         -20.292, 148.703, 0.050),
    # Townsville
    ("4810", "Townsville",           -19.260, 146.820, 0.050),
    ("4811", "Kirwan",               -19.322, 146.751, 0.035),
    ("4814", "Hyde Park",            -19.307, 146.788, 0.030),
    ("4817", "Garbutt",              -19.284, 146.764, 0.030),
    ("4819", "Magnetic Island",      -19.139, 146.872, 0.040),
    ("4820", "Charters Towers",      -20.074, 146.263, 0.080),
    # Mount Isa / western QLD
    ("4825", "Mount Isa",            -20.730, 139.492, 0.150),
    ("4830", "Cloncurry",            -20.710, 140.512, 0.150),
    ("4720", "Longreach",            -23.440, 144.252, 0.200),
    ("4730", "Charleville",          -26.402, 146.242, 0.200),
    # Cairns / Far North
    ("4870", "Cairns",               -16.920, 145.770, 0.050),
    ("4871", "Cairns North",         -16.899, 145.762, 0.035),
    ("4872", "Atherton",             -17.267, 145.477, 0.060),
    ("4873", "Mossman",              -16.462, 145.372, 0.060),
    ("4874", "Cooktown",             -15.468, 145.249, 0.100),
    ("4875", "Weipa",                -12.661, 141.863, 0.150),
    ("4880", "Mareeba",              -17.001, 145.428, 0.060),
    # Ingham / north coast
    ("4850", "Ingham",               -18.650, 146.160, 0.060),
    ("4869", "Innisfail",            -17.523, 146.028, 0.055),
    # Toowoomba / Darling Downs
    ("4350", "Toowoomba",            -27.550, 151.952, 0.045),
    ("4370", "Warwick",              -28.213, 152.037, 0.060),
    ("4390", "Goondiwindi",          -28.550, 150.310, 0.100),
    ("4400", "Dalby",                -27.183, 151.263, 0.070),
    ("4410", "Roma",                 -26.571, 148.793, 0.100),
    # Rockhampton region / central west
    ("4701", "Emerald",              -23.524, 148.168, 0.100),
    ("4721", "Barcaldine",           -23.558, 145.290, 0.200),
    # Normanton / Gulf
    ("4890", "Normanton",            -17.678, 141.078, 0.200),
]


def _seifa_from_coords(lat: float, lon: float, rng: np.random.Generator) -> dict:
    """
    Produce synthetic but spatially-plausible SEIFA scores.
    Urban coastal = higher deciles; remote outback = lower deciles.
    """
    # Remoteness proxy: distance from Brisbane (roughly)
    dist = ((lat - (-27.47)) ** 2 + (lon - 153.02) ** 2) ** 0.5
    # Coastal bonus: closer to east coast = more advantaged
    coastal = max(0.0, 1.0 - abs(lon - 153.0) / 5.0)

    base = 8.0 - dist * 1.8 + coastal * 2.0 + rng.normal(0, 1.2)
    decile = int(np.clip(round(base), 1, 10))

    score = 900 + (decile - 5) * 30 + int(rng.normal(0, 15))
    return {
        "IRSD_SCORE": score,
        "IRSD_DECILE_AUST": decile,
        "IRSD_RANK_AUST": rng.integers(1, 2700),
        "IRSAD_SCORE": score + int(rng.normal(0, 10)),
        "IRSAD_DECILE_AUST": int(np.clip(decile + rng.integers(-1, 2), 1, 10)),
        "IRSAD_RANK_AUST": rng.integers(1, 2700),
        "IEO_SCORE": score + int(rng.normal(0, 20)),
        "IEO_DECILE_AUST": int(np.clip(decile + rng.integers(-1, 2), 1, 10)),
        "IEO_RANK_AUST": rng.integers(1, 2700),
        "IER_SCORE": score + int(rng.normal(0, 20)),
        "IER_DECILE_AUST": int(np.clip(decile + rng.integers(-1, 2), 1, 10)),
        "IER_RANK_AUST": rng.integers(1, 2700),
        "POA_NAME_2021": "",
        "USUAL_RESIDENT_POP": int(max(200, rng.normal(8000 if decile >= 6 else 2000, 3000))),
    }


def generate_demo_data() -> gpd.GeoDataFrame:
    """Build a GeoDataFrame with synthetic QLD postcode data, same schema as the real pipeline."""
    rng = np.random.default_rng(42)
    records = []
    for poa, suburb, lat, lon, half in _QLD_POSTCODES:
        geom = box(lon - half, lat - half * 0.7, lon + half, lat + half * 0.7)
        seifa = _seifa_from_coords(lat, lon, rng)
        irsd_d = seifa["IRSD_DECILE_AUST"]

        # Unemployment inversely correlated with SEIFA decile
        unemp = float(np.clip(rng.normal(18 - irsd_d * 1.4, 2.5), 2.0, 25.0))
        lf_tot = int(rng.normal(4000 if irsd_d >= 6 else 1500, 1000))
        unemp_tot = int(lf_tot * unemp / 100)

        # Income positively correlated with SEIFA
        median_inc = int(np.clip(rng.normal(600 + irsd_d * 70, 80), 300, 2000))
        median_rent = int(np.clip(rng.normal(350 + irsd_d * 25, 50), 150, 700))
        median_age = int(np.clip(rng.normal(36 + irsd_d * 0.5, 4), 24, 55))

        area_km2 = round((half * 111) ** 2 * 2, 1)

        records.append({
            "POA_CODE21": poa,
            "POA_NAME21": suburb,
            "AREASQKM21": area_km2,
            "geometry": geom,
            # SEIFA fields
            **{k: v for k, v in seifa.items() if k != "POA_NAME_2021"},
            # Labour force
            "P_LF_Tot": lf_tot,
            "P_Unemp_Tot": unemp_tot,
            "unemployment_rate": round(unemp, 1),
            # Medians
            "Median_age_persons": median_age,
            "Median_tot_prsnl_inc_weekly": median_inc,
            "Median_rent_weekly": median_rent,
            "Median_tot_hhd_inc_weekly": int(median_inc * 1.6),
            "Average_household_size": round(rng.uniform(1.8, 3.2), 1),
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    print(f"  Demo dataset: {len(gdf)} synthetic QLD postcodes")
    return gdf


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_map(gdf: gpd.GeoDataFrame, demo: bool = False) -> folium.Map:
    m = folium.Map(
        location=[-22.0, 144.0],
        zoom_start=5,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    if demo:
        folium.map.Marker(
            [-10.5, 138.5],
            icon=folium.DivIcon(
                html='<div style="background:rgba(255,200,0,0.85);padding:6px 10px;'
                     'border-radius:4px;font-size:12px;white-space:nowrap;">'
                     '⚠ Demo mode — synthetic data only</div>',
                icon_size=(260, 36),
            ),
        ).add_to(m)

    geojson_str = gdf.to_json()

    for layer in LAYERS:
        valid_data = gdf[["POA_CODE21", layer["col"]]].dropna()
        cp = folium.Choropleth(
            geo_data=geojson_str,
            name=layer["name"],
            data=valid_data,
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
                fields=present,
                aliases=aliases,
                localize=True,
                sticky=False,
                labels=True,
                style="font-size:12px;",
            )
        )

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build QLD Census 2021 postcode choropleth map.")
    parser.add_argument(
        "--demo", action="store_true",
        help="Skip downloads and use built-in synthetic data (useful without internet access).",
    )
    args = parser.parse_args()

    t0 = time.time()
    demo_mode = args.demo

    if not demo_mode:
        try:
            for d in [SEIFA_DIR, BOUNDARY_DIR, DATAPACK_DIR]:
                d.mkdir(parents=True, exist_ok=True)

            print("Downloading SEIFA 2021 ...")
            seifa_path = download_file(SEIFA_URL, SEIFA_DIR / "Postal_Area_Indexes_SEIFA_2021.xlsx")

            print("Downloading POA boundaries ...")
            boundary_zip = download_file(BOUNDARY_URL, BOUNDARY_DIR / "POA_2021_AUST_SHP_GDA2020.zip")

            print("Downloading Census DataPack (QLD POA) ...")
            datapack_zip = download_file(DATAPACK_URL, DATAPACK_DIR / "2021_GCP_POA_for_QLD_short-header.zip")

            print("Loading SEIFA ...")
            df_seifa = load_seifa(seifa_path)
            print(f"  SEIFA rows: {len(df_seifa)}")

            print("Extracting DataPack CSVs ...")
            g17a_path, g02_path = extract_datapack_csvs(datapack_zip, DATAPACK_DIR)

            print("Loading labour force data ...")
            df_lf = load_labour_force(g17a_path)
            print(f"  Labour force rows: {len(df_lf)}")

            print("Loading medians data ...")
            df_med = load_medians(g02_path)
            print(f"  Medians rows: {len(df_med)}")

            print("Loading POA boundaries ...")
            extract_dir = BOUNDARY_DIR / "POA_2021_AUST_GDA2020"
            gdf = load_boundaries(boundary_zip, extract_dir)
            print(f"  Boundary features: {len(gdf)}")

            print("Merging datasets ...")
            gdf_final = merge_datasets(gdf, df_seifa, df_lf, df_med)

        except RuntimeError as exc:
            print(f"\n[!] Download failed: {exc}", file=sys.stderr)
            print("[!] Falling back to demo mode with synthetic data.\n", file=sys.stderr)
            demo_mode = True

    if demo_mode:
        print("Generating synthetic demo data ...")
        gdf_final = generate_demo_data()

    print("Building choropleth map ...")
    m = build_map(gdf_final, demo=demo_mode)
    m.save(str(OUTPUT_HTML))

    elapsed = time.time() - t0
    print(f"\nDone!  {OUTPUT_HTML}")
    print(f"Size:  {OUTPUT_HTML.stat().st_size / 1e6:.1f} MB")
    print(f"Time:  {elapsed:.1f}s")
    if demo_mode:
        print("\nNote: map uses synthetic data. Run without --demo (with internet access)")
        print("      to download real ABS Census 2021 data from abs.gov.au.")


if __name__ == "__main__":
    main()
