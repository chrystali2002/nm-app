import ee
import json
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from datetime import date
import socket
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from branca.element import MacroElement, Template

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Atmospheric Gas Comparison (Sentinel-5P) for Africa / African Countries",
    page_icon="🌍",
    layout="wide"
)

st.image("https://github.com/chrystali2002.png", width=100)
st.header("Timeframe Comparison of Atmospheric Gases over Africa or Any African Country", divider="rainbow")

st.write("""
This dataset provides near real-time high-resolution imagery of UV Aerosol Index, concentrations of Carbon monoxide (CO), water vapor, and Formaldehyde. 
Other important greenhouse gases such as the total, tropospheric, and stratospheric nitrogen dioxide (NO₂), total atmospheric column ozone (O₃), atmospheric sulphur dioxide (SO₂), and atmospheric methane (CH₄) are included in this application for visualization.

The source of this dataset is obtained from the 
    <a href="https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p" target="_blank">
        <b>Earth Engine Data Catalog (Sentinel‑5P)</b>
    </a>, 
    and the details of these individual gases are available in the catalog.
    
This dataset is sourced from the Earth Engine Data Catalog (Sentinel‑5P), and details for individual gases are available in the catalog.
This application enables users to visually compare atmospheric gas concentrations from Sentinel-5P
for **all Africa** or any selected **African country** across two selected timeframes.

The left and right panels use a **shared dynamic color scale** for direct comparison,
while the difference map shows **Right − Left** using its own symmetric blue-white-red scale.
""",
    unsafe_allow_html=True)

# ---------------------------------------------------------
# EARTH ENGINE INITIALIZATION
# ---------------------------------------------------------
@st.cache_resource
def initialize_earth_engine():
    try:
        service_account_info = json.loads(st.secrets["EE_CREDENTIAL_JSON"])
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info["client_email"],
            key_data=json.dumps(service_account_info)
        )
        ee.Initialize(credentials)
        ee.data._credentials = credentials
        socket.setdefaulttimeout(60)
        return True
    except Exception as e:
        st.error(f"Earth Engine initialization failed: {e}")
        return False

if not initialize_earth_engine():
    st.stop()

# ---------------------------------------------------------
# AFRICA REGION DEFINITIONS
# ---------------------------------------------------------
AFRICA_COUNTRIES = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros",
    "Congo", "Democratic Republic of the Congo", "Djibouti", "Egypt",
    "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Côte d'Ivoire", "Kenya",
    "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
    "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger",
    "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles",
    "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
    "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
]

REGION_OPTIONS = ["All Africa"] + AFRICA_COUNTRIES

REGION_CENTER_ZOOM = {
    "All Africa": ([2.5, 20.0], 3),
    "Algeria": ([28.0, 2.6], 5),
    "Angola": ([-11.5, 17.9], 5),
    "Benin": ([9.5, 2.3], 6),
    "Botswana": ([-22.3, 24.7], 6),
    "Burkina Faso": ([12.3, -1.6], 6),
    "Burundi": ([-3.4, 29.9], 7),
    "Cabo Verde": ([16.0, -24.0], 7),
    "Cameroon": ([5.7, 12.7], 6),
    "Central African Republic": ([6.6, 20.9], 6),
    "Chad": ([15.3, 18.7], 5),
    "Comoros": ([-11.9, 43.3], 8),
    "Congo": ([-0.7, 15.2], 6),
    "Democratic Republic of the Congo": ([-2.8, 23.7], 5),
    "Djibouti": ([11.8, 42.6], 8),
    "Egypt": ([26.8, 30.8], 5),
    "Equatorial Guinea": ([1.6, 10.5], 7),
    "Eritrea": ([15.3, 39.3], 6),
    "Eswatini": ([-26.5, 31.5], 8),
    "Ethiopia": ([9.1, 40.5], 6),
    "Gabon": ([-0.6, 11.8], 6),
    "Gambia": ([13.4, -15.4], 8),
    "Ghana": ([7.9, -1.0], 6),
    "Guinea": ([10.4, -10.9], 6),
    "Guinea-Bissau": ([12.0, -15.0], 7),
    "Côte d'Ivoire": ([7.6, -5.5], 6),
    "Kenya": ([0.2, 37.9], 6),
    "Lesotho": ([-29.6, 28.2], 8),
    "Liberia": ([6.4, -9.4], 7),
    "Libya": ([27.0, 18.0], 5),
    "Madagascar": ([-19.0, 46.7], 5),
    "Malawi": ([-13.2, 34.3], 6),
    "Mali": ([17.3, -3.5], 5),
    "Mauritania": ([20.3, -10.3], 5),
    "Mauritius": ([-20.2, 57.5], 9),
    "Morocco": ([31.8, -6.0], 6),
    "Mozambique": ([-18.7, 35.5], 5),
    "Namibia": ([-22.1, 17.2], 6),
    "Niger": ([17.6, 9.4], 5),
    "Nigeria": ([9.1, 8.7], 6),
    "Rwanda": ([-1.9, 29.9], 8),
    "Sao Tome and Principe": ([0.3, 6.7], 9),
    "Senegal": ([14.5, -14.5], 6),
    "Seychelles": ([-4.7, 55.5], 9),
    "Sierra Leone": ([8.5, -11.8], 7),
    "Somalia": ([5.2, 46.2], 5),
    "South Africa": ([-29.0, 24.0], 5),
    "South Sudan": ([7.8, 30.0], 6),
    "Sudan": ([15.6, 30.5], 5),
    "Tanzania": ([-6.3, 34.8], 6),
    "Togo": ([8.6, 1.1], 7),
    "Tunisia": ([34.0, 9.5], 6),
    "Uganda": ([1.4, 32.3], 7),
    "Zambia": ([-13.2, 27.8], 6),
    "Zimbabwe": ([-19.0, 29.2], 6),
}

# ---------------------------------------------------------
# REGION GEOMETRY
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_region_geometry(region_name):
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

    if region_name == "All Africa":
        africa_fc = countries.filter(ee.Filter.eq("wld_rgn", "Africa"))
        return africa_fc.geometry()

    # Match by country_na
    feature = countries.filter(ee.Filter.eq("country_na", region_name)).first()
    return ee.Feature(feature).geometry()

# ---------------------------------------------------------
# UI CONTROLS
# ---------------------------------------------------------
region_name = st.selectbox(
    "Choose region:",
    REGION_OPTIONS,
    index=0
)

region_geom = get_region_geometry(region_name)
map_center, zoom_level = REGION_CENTER_ZOOM.get(region_name, ([2.5, 20.0], 3))

gas = st.selectbox(
    "Select the gas:",
    [
        "Concentrations of Carbon monoxide (CO)",
        "Concentrations of water vapor",
        "UV Aerosol Index",
        "Concentrations of Formaldehyde",
        "Concentrations of total, tropospheric, and stratospheric nitrogen dioxide",
        "Concentrations of total atmospheric column ozone",
        "Concentrations of atmospheric sulphur dioxide (SO₂)",
        "Concentrations of atmospheric methane (CH₄)"
    ]
)

month_dict = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
month_names = list(month_dict.keys())

col1, col2 = st.columns(2)

with col1:
    st.write("##### Left Panel")
    month_l = st.selectbox("Month:", month_names, key="m1")
    year_l = st.slider("Year:", 2018, 2025, 2021, key="y1")

with col2:
    st.write("##### Right Panel")
    month_r = st.selectbox("Month:", month_names, key="m2")
    year_r = st.slider("Year:", 2018, 2025, 2023, key="y2")

opacity = st.slider("Overlay Opacity", 0.1, 1.0, 0.8)

# ---------------------------------------------------------
# GAS CONFIG
# ---------------------------------------------------------
gas_dict = {
    "Concentrations of Carbon monoxide (CO)": {
        "offl_col": "COPERNICUS/S5P/OFFL/L3_CO",
        "nrti_col": "COPERNICUS/S5P/NRTI/L3_CO",
        "band": "CO_column_number_density",
        "unit": "mol/m²"
    },
    "Concentrations of water vapor": {
        "col": "COPERNICUS/S5P/NRTI/L3_CO",
        "band": "H2O_column_number_density",
        "unit": "mol/m²"
    },
    "UV Aerosol Index": {
        "col": "COPERNICUS/S5P/NRTI/L3_AER_AI",
        "band": "absorbing_aerosol_index",
        "unit": "unitless"
    },
    "Concentrations of Formaldehyde": {
        "col": "COPERNICUS/S5P/NRTI/L3_HCHO",
        "band": "tropospheric_HCHO_column_number_density",
        "unit": "mol/m²"
    },
    "Concentrations of total, tropospheric, and stratospheric nitrogen dioxide": {
        "col": "COPERNICUS/S5P/NRTI/L3_NO2",
        "band": "NO2_column_number_density",
        "unit": "mol/m²"
    },
    "Concentrations of total atmospheric column ozone": {
        "col": "COPERNICUS/S5P/NRTI/L3_O3",
        "band": "O3_column_number_density",
        "unit": "mol/m²"
    },
    "Concentrations of atmospheric sulphur dioxide (SO₂)": {
        "col": "COPERNICUS/S5P/NRTI/L3_SO2",
        "band": "SO2_column_number_density",
        "unit": "mol/m²"
    },
    "Concentrations of atmospheric methane (CH₄)": {
        "col": "COPERNICUS/S5P/OFFL/L3_CH4",
        "band": "CH4_column_volume_mixing_ratio_dry_air",
        "unit": "ppbv"
    }
}

FAST_RANGES = {
    "CO_column_number_density": (0.02, 0.05),
    "H2O_column_number_density": (0.0, 5.0e4),
    "absorbing_aerosol_index": (0.0, 7.0),
    "tropospheric_HCHO_column_number_density": (0.0, 5.0e-4),
    "NO2_column_number_density": (0.0, 3.0e-4),
    "O3_column_number_density": (0.0, 0.45),
    "SO2_column_number_density": (0.0, 0.03),
    "CH4_column_volume_mixing_ratio_dry_air": (1680.0, 1980.0),
}

SHARED_PALETTE = ["black", "blue", "purple", "cyan", "green", "yellow", "red"]
DIFF_PALETTE = ["blue", "white", "red"]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def fmt_tick(x):
    if x is None:
        return "0"
    try:
        x = float(x)
    except Exception:
        return str(x)

    if np.isnan(x) or np.isinf(x):
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    if abs(x) >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2e}"

def add_html_legend(map_obj, title, vmin, vmax, colors, position="bottomleft"):
    pos_styles = {
        "bottomleft": "bottom: 25px; left: 25px;",
        "bottomright": "bottom: 25px; right: 25px;",
        "bottomcenter": "bottom: 25px; left: 50%; transform: translateX(-50%);",
    }
    style_pos = pos_styles.get(position, pos_styles["bottomleft"])
    gradient = ", ".join(colors)
    mid = (vmin + vmax) / 2

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        {style_pos}
        z-index: 999999;
        background-color: rgba(255, 255, 255, 0.92);
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        min-width: 250px;
    ">
        <div style="font-weight: 700; margin-bottom: 8px; line-height: 1.2;">
            {title}
        </div>
        <div style="
            height: 14px;
            width: 100%;
            border-radius: 4px;
            border: 1px solid #666;
            background: linear-gradient(to right, {gradient});
            margin-bottom: 6px;
        "></div>
        <div style="
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #222;
        ">
            <span>{fmt_tick(vmin)}</span>
            <span>{fmt_tick(mid)}</span>
            <span>{fmt_tick(vmax)}</span>
        </div>
    </div>
    {{% endmacro %}}
    """

    macro = MacroElement()
    macro._template = Template(legend_html)
    map_obj.get_root().add_child(macro)

def ee_month_range(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return start, end

def month_end_python(year, month):
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)

def infer_stream_from_id(col_id: str) -> str:
    if "/OFFL/" in col_id:
        return "OFFL"
    if "/NRTI/" in col_id:
        return "NRTI"
    return "UNKNOWN"

def choose_co_stream(year, month):
    end = month_end_python(year, month)
    days_ago = (date.today() - end).days
    return "NRTI" if days_ago < 14 else "OFFL"

def build_image(gas_key, year, month, geom):
    start, end = ee_month_range(year, month)
    cfg = gas_dict[gas_key]
    band = cfg["band"]

    if "offl_col" in cfg and "nrti_col" in cfg:
        stream = choose_co_stream(year, month)
        col_id = cfg["nrti_col"] if stream == "NRTI" else cfg["offl_col"]
    else:
        col_id = cfg["col"]
        stream = infer_stream_from_id(col_id)

    col = (
        ee.ImageCollection(col_id)
        .select(band)
        .filterDate(start, end)
        .filterBounds(geom)
    )

    img = ee.Image(
        ee.Algorithms.If(
            col.size().gt(0),
            col.median(),
            ee.Image.constant(0).rename(band)
        )
    ).clip(geom).float()

    return img, stream, col_id

def get_percentile_range(img, band, geom, fallback, p_low=2, p_high=98, scale=50000):
    try:
        stats = img.reduceRegion(
            reducer=ee.Reducer.percentile([p_low, p_high]),
            geometry=geom,
            scale=scale,
            bestEffort=True,
            maxPixels=1e7
        ).getInfo()

        low_key = f"{band}_p{p_low}"
        high_key = f"{band}_p{p_high}"

        vmin = stats.get(low_key)
        vmax = stats.get(high_key)

        if vmin is None or vmax is None:
            return fallback

        vmin = float(vmin)
        vmax = float(vmax)

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return fallback

        if vmin == vmax:
            eps = max(abs(vmin) * 0.05, 1e-6)
            return vmin - eps, vmax + eps

        pad = 0.05 * (vmax - vmin)
        return vmin - pad, vmax + pad

    except Exception:
        return fallback

def get_percentile_diff_range(img_diff, band, geom, fallback_abs, p_low=2, p_high=98, scale=50000):
    try:
        stats = img_diff.reduceRegion(
            reducer=ee.Reducer.percentile([p_low, p_high]),
            geometry=geom,
            scale=scale,
            bestEffort=True,
            maxPixels=1e7
        ).getInfo()

        low_key = f"{band}_p{p_low}"
        high_key = f"{band}_p{p_high}"

        vlow = stats.get(low_key)
        vhigh = stats.get(high_key)

        if vlow is None or vhigh is None:
            return -fallback_abs, fallback_abs

        vlow = float(vlow)
        vhigh = float(vhigh)

        if not np.isfinite(vlow) or not np.isfinite(vhigh):
            return -fallback_abs, fallback_abs

        max_abs = max(abs(vlow), abs(vhigh))
        max_abs = max(max_abs * 1.05, 1e-6)

        return -max_abs, max_abs

    except Exception:
        return -fallback_abs, fallback_abs

# ---------------------------------------------------------
# CACHED EE PIPELINE
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception)
)
def compute_tiles_and_meta(
    gas_key: str,
    region_name: str,
    year_l: int,
    month_l_num: int,
    year_r: int,
    month_r_num: int,
):
    geom = get_region_geometry(region_name)

    band = gas_dict[gas_key]["band"]
    unit = gas_dict[gas_key]["unit"]

    img_left, stream_left, col_left = build_image(gas_key, year_l, month_l_num, geom)
    img_right, stream_right, col_right = build_image(gas_key, year_r, month_r_num, geom)
    img_diff = img_right.subtract(img_left).rename(band)

    fallback_min, fallback_max = FAST_RANGES.get(band, (0.0, 1.0))
    fallback_diff = max((fallback_max - fallback_min) * 0.3, 1e-6)

    left_leg_min, left_leg_max = get_percentile_range(
        img_left, band, geom, (fallback_min, fallback_max),
        p_low=2, p_high=98, scale=50000
    )
    right_leg_min, right_leg_max = get_percentile_range(
        img_right, band, geom, (fallback_min, fallback_max),
        p_low=2, p_high=98, scale=50000
    )

    shared_leg_min = min(left_leg_min, right_leg_min)
    shared_leg_max = max(left_leg_max, right_leg_max)

    diff_leg_min, diff_leg_max = get_percentile_diff_range(
        img_diff, band, geom, fallback_diff,
        p_low=2, p_high=98, scale=50000
    )

    viz_left = {
        "min": shared_leg_min,
        "max": shared_leg_max,
        "palette": SHARED_PALETTE,
    }
    viz_right = {
        "min": shared_leg_min,
        "max": shared_leg_max,
        "palette": SHARED_PALETTE,
    }
    viz_diff = {
        "min": diff_leg_min,
        "max": diff_leg_max,
        "palette": DIFF_PALETTE,
    }

    left_tiles = img_left.getMapId(viz_left)["tile_fetcher"].url_format
    right_tiles = img_right.getMapId(viz_right)["tile_fetcher"].url_format
    diff_tiles = img_diff.getMapId(viz_diff)["tile_fetcher"].url_format

    return {
        "band": band,
        "unit": unit,
        "left_tiles": left_tiles,
        "right_tiles": right_tiles,
        "diff_tiles": diff_tiles,
        "left_min": left_leg_min,
        "left_max": left_leg_max,
        "right_min": right_leg_min,
        "right_max": right_leg_max,
        "shared_min": shared_leg_min,
        "shared_max": shared_leg_max,
        "diff_min": diff_leg_min,
        "diff_max": diff_leg_max,
        "stream_left": stream_left,
        "stream_right": stream_right,
        "col_left": col_left,
        "col_right": col_right,
    }

# ---------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------
try:
    with st.spinner(f"Preparing map layers for {region_name}..."):
        meta = compute_tiles_and_meta(
            gas,
            region_name,
            year_l, month_dict[month_l],
            year_r, month_dict[month_r],
        )

    with st.expander("🔍 Data Validation", expanded=True):
        st.write("### Panel Statistics")
        st.write(f"**Selected region:** {region_name}")
        st.write(f"**Left panel raw range:** {meta['left_min']:.6f} to {meta['left_max']:.6f}")
        st.write(f"**Right panel raw range:** {meta['right_min']:.6f} to {meta['right_max']:.6f}")
        st.write(f"**Shared comparison range:** {meta['shared_min']:.6f} to {meta['shared_max']:.6f}")
        st.write(f"**Difference range:** {meta['diff_min']:.6f} to {meta['diff_max']:.6f}")

except Exception as e:
    st.error(f"Error preparing map layers: {str(e)}")
    st.stop()

unit = meta["unit"]
stream_left = meta["stream_left"]
stream_right = meta["stream_right"]

# ---------------------------------------------------------
# SIDE-BY-SIDE MAPS
# ---------------------------------------------------------
st.subheader(f"Comparative View — {region_name}")

label_col1, label_col2 = st.columns(2)
with label_col1:
    st.markdown(f"<h3 style='text-align: center;'>LEFT PANEL: {month_l} {year_l}</h3>", unsafe_allow_html=True)
with label_col2:
    st.markdown(f"<h3 style='text-align: center;'>RIGHT PANEL: {month_r} {year_r}</h3>", unsafe_allow_html=True)

map_col1, map_col2 = st.columns(2)

with map_col1:
    left_map = folium.Map(location=map_center, zoom_start=zoom_level, control_scale=True)

    folium.TileLayer(
        tiles=meta["left_tiles"],
        name=f"{month_l} {year_l}",
        attr="Sentinel-5P • Google Earth Engine",
        opacity=opacity
    ).add_to(left_map)

    folium.LayerControl().add_to(left_map)

    add_html_legend(
        left_map,
        title=f"{gas}<br>{region_name}<br>{month_l} {year_l} ({unit})<br><span style='font-weight:400;'>Shared comparison scale</span>",
        vmin=meta["shared_min"],
        vmax=meta["shared_max"],
        colors=SHARED_PALETTE,
        position="bottomleft"
    )

    st_folium(
        left_map,
        height=550,
        width=500,
        key=f"left_map_{region_name}_{gas}_{month_l}_{year_l}_{opacity}",
        returned_objects=[]
    )

with map_col2:
    right_map = folium.Map(location=map_center, zoom_start=zoom_level, control_scale=True)

    folium.TileLayer(
        tiles=meta["right_tiles"],
        name=f"{month_r} {year_r}",
        attr="Sentinel-5P • Google Earth Engine",
        opacity=opacity
    ).add_to(right_map)

    folium.LayerControl().add_to(right_map)

    add_html_legend(
        right_map,
        title=f"{gas}<br>{region_name}<br>{month_r} {year_r} ({unit})<br><span style='font-weight:400;'>Shared comparison scale</span>",
        vmin=meta["shared_min"],
        vmax=meta["shared_max"],
        colors=SHARED_PALETTE,
        position="bottomright"
    )

    st_folium(
        right_map,
        height=550,
        width=500,
        key=f"right_map_{region_name}_{gas}_{month_r}_{year_r}_{opacity}",
        returned_objects=[]
    )

# ---------------------------------------------------------
# DIFFERENCE MAP
# ---------------------------------------------------------
st.subheader(f"Difference Map for {region_name} (Right − Left)")

diff_map = folium.Map(location=map_center, zoom_start=zoom_level, control_scale=True)

folium.TileLayer(
    tiles=meta["diff_tiles"],
    name=f"Difference ({month_r} {year_r} - {month_l} {year_l})",
    attr="Sentinel-5P • Google Earth Engine",
    opacity=opacity
).add_to(diff_map)

folium.LayerControl().add_to(diff_map)

add_html_legend(
    diff_map,
    title=f"{gas} Difference<br>{region_name}<br>({month_r} {year_r} − {month_l} {year_l}) ({unit})",
    vmin=meta["diff_min"],
    vmax=meta["diff_max"],
    colors=DIFF_PALETTE,
    position="bottomcenter"
)

st_folium(
    diff_map,
    height=500,
    width=1100,
    key=f"diff_map_{region_name}_{gas}_{month_l}_{year_l}_{month_r}_{year_r}_{opacity}",
    returned_objects=[]
)

# ---------------------------------------------------------
# DATA QUALITY NOTES
# ---------------------------------------------------------
with st.expander("📊 Data Quality Notes", expanded=False):
    st.markdown(f"""
**Selected region**
- `{region_name}`

**Data Sources**
- Sentinel-5P Level 3 products from the Copernicus Programme
- OFFL (Offline) and NRTI (Near Real-Time) collections

**Current stream selection**
- Left panel stream: `{stream_left}`
- Right panel stream: `{stream_right}`

**Left vs Right comparison**
- The left and right panels use one **shared color scale**
- That shared scale is based on the combined percentile range from the two selected periods
- This makes visual comparison between the two panels scientifically consistent

**Difference map**
- Shows absolute difference: **Right panel − Left panel**
- 🔴 **Red** = Increase
- ⚪ **White** = No change
- 🔵 **Blue** = Decrease

**Units**
- `mol/m²`: moles per square meter
- `ppbv`: parts per billion by volume
- `unitless`: no physical units
""")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption("Data provided by Google Earth Engine and Copernicus Sentinel-5P")
