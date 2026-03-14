import ee
import json
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import socket
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from branca.element import MacroElement, Template

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Atmospheric Gas Comparison (Sentinel-5P) for U.S. States",
    page_icon="🌎",
    layout="wide"
)

st.image("https://github.com/chrystali2002.png", width=100)
st.header("Timeframe Comparison of Atmospheric Gases over U.S. States", divider="rainbow")

st.write("""
This application enables users to visually compare atmospheric gas concentrations from Sentinel-5P
for any selected U.S. state across two selected timeframes using a side-by-side interface,
with a dynamic difference map showing changes (Right − Left).

Both left and right panels use a **shared dynamic color scale** for direct comparison.
""")

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
# STATE DEFINITIONS
# ---------------------------------------------------------
STATE_NAMES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

STATE_CENTER_ZOOM = {
    "Alabama": ([32.8, -86.8], 6),
    "Alaska": ([64.2, -152.0], 3),
    "Arizona": ([34.2, -111.7], 6),
    "Arkansas": ([34.9, -92.4], 6),
    "California": ([37.2, -119.5], 5),
    "Colorado": ([39.0, -105.5], 6),
    "Connecticut": ([41.6, -72.7], 7),
    "Delaware": ([39.0, -75.5], 8),
    "Florida": ([27.8, -81.7], 6),
    "Georgia": ([32.7, -83.4], 6),
    "Hawaii": ([20.8, -157.5], 7),
    "Idaho": ([44.2, -114.6], 6),
    "Illinois": ([40.0, -89.2], 6),
    "Indiana": ([39.9, -86.3], 6),
    "Iowa": ([42.1, -93.5], 6),
    "Kansas": ([38.5, -98.0], 6),
    "Kentucky": ([37.8, -85.8], 6),
    "Louisiana": ([31.0, -92.0], 6),
    "Maine": ([45.2, -69.0], 6),
    "Maryland": ([39.0, -76.7], 7),
    "Massachusetts": ([42.3, -71.8], 7),
    "Michigan": ([44.3, -85.5], 6),
    "Minnesota": ([46.0, -94.0], 6),
    "Mississippi": ([32.7, -89.7], 6),
    "Missouri": ([38.5, -92.5], 6),
    "Montana": ([46.9, -110.0], 5),
    "Nebraska": ([41.5, -99.8], 6),
    "Nevada": ([39.3, -116.6], 6),
    "New Hampshire": ([43.8, -71.6], 7),
    "New Jersey": ([40.1, -74.7], 7),
    "New Mexico": ([34.5, -106.0], 6),
    "New York": ([42.9, -75.5], 6),
    "North Carolina": ([35.5, -79.4], 6),
    "North Dakota": ([47.5, -100.5], 6),
    "Ohio": ([40.3, -82.8], 6),
    "Oklahoma": ([35.6, -97.5], 6),
    "Oregon": ([44.0, -120.5], 6),
    "Pennsylvania": ([41.0, -77.6], 6),
    "Rhode Island": ([41.7, -71.5], 8),
    "South Carolina": ([33.8, -80.9], 6),
    "South Dakota": ([44.4, -100.2], 6),
    "Tennessee": ([35.8, -86.4], 6),
    "Texas": ([31.0, -99.0], 5),
    "Utah": ([39.3, -111.7], 6),
    "Vermont": ([44.1, -72.7], 7),
    "Virginia": ([37.5, -78.8], 6),
    "Washington": ([47.4, -120.7], 6),
    "West Virginia": ([38.6, -80.6], 7),
    "Wisconsin": ([44.5, -89.5], 6),
    "Wyoming": ([43.0, -107.5], 6),
}

# ---------------------------------------------------------
# GET STATE GEOMETRY
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_state_geometry(state_name):
    states_fc = ee.FeatureCollection("TIGER/2018/States")
    feature = states_fc.filter(ee.Filter.eq("NAME", state_name)).first()
    geom = ee.Feature(feature).geometry()
    return geom


# ---------------------------------------------------------
# UI CONTROLS
# ---------------------------------------------------------
state_name = st.selectbox(
    "Select U.S. state:",
    STATE_NAMES,
    index=STATE_NAMES.index("New Mexico")
)

state_geom = get_state_geometry(state_name)
map_center, zoom_level = STATE_CENTER_ZOOM.get(state_name, ([39.5, -98.35], 5))

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

# exact agreement mode
fast_mode = False

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
        "topright": "top: 25px; right: 25px;",
        "topleft": "top: 25px; left: 25px;"
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
        min-width: 240px;
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


def build_image(gas_key, year, month, region_geom):
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
        .filterBounds(region_geom)
    )

    img = ee.Image(
        ee.Algorithms.If(
            col.size().gt(0),
            col.median(),
            ee.Image.constant(0).rename(band)
        )
    ).clip(region_geom).float()

    return img, stream, col_id


def get_percentile_range(img, band, region_geom, fallback, p_low=2, p_high=98, scale=50000):
    try:
        stats = img.reduceRegion(
            reducer=ee.Reducer.percentile([p_low, p_high]),
            geometry=region_geom,
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


def get_percentile_diff_range(img_diff, band, region_geom, fallback_abs, p_low=2, p_high=98, scale=50000):
    try:
        stats = img_diff.reduceRegion(
            reducer=ee.Reducer.percentile([p_low, p_high]),
            geometry=region_geom,
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
    state_name: str,
    year_l: int,
    month_l_num: int,
    year_r: int,
    month_r_num: int,
    fast_mode: bool
):
    region_geom = get_state_geometry(state_name)

    band = gas_dict[gas_key]["band"]
    unit = gas_dict[gas_key]["unit"]

    img_left, stream_left, col_left = build_image(gas_key, year_l, month_l_num, region_geom)
    img_right, stream_right, col_right = build_image(gas_key, year_r, month_r_num, region_geom)
    img_diff = img_right.subtract(img_left).rename(band)

    fallback_min, fallback_max = FAST_RANGES.get(band, (0.0, 1.0))
    fallback_diff = max((fallback_max - fallback_min) * 0.3, 1e-6)

    left_leg_min, left_leg_max = get_percentile_range(
        img_left, band, region_geom, (fallback_min, fallback_max),
        p_low=2, p_high=98, scale=50000
    )
    right_leg_min, right_leg_max = get_percentile_range(
        img_right, band, region_geom, (fallback_min, fallback_max),
        p_low=2, p_high=98, scale=50000
    )

    shared_leg_min = min(left_leg_min, right_leg_min)
    shared_leg_max = max(left_leg_max, right_leg_max)

    diff_leg_min, diff_leg_max = get_percentile_diff_range(
        img_diff, band, region_geom, fallback_diff,
        p_low=2, p_high=98, scale=50000
    )

    # exact agreement mode: tiles use same ranges as legends
    shared_tile_min, shared_tile_max = shared_leg_min, shared_leg_max
    diff_tile_min, diff_tile_max = diff_leg_min, diff_leg_max

    viz_left = {
        "min": shared_tile_min,
        "max": shared_tile_max,
        "palette": SHARED_PALETTE,
    }
    viz_right = {
        "min": shared_tile_min,
        "max": shared_tile_max,
        "palette": SHARED_PALETTE,
    }
    viz_diff = {
        "min": diff_tile_min,
        "max": diff_tile_max,
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
    with st.spinner(f"Preparing map layers for {state_name}..."):
        meta = compute_tiles_and_meta(
            gas,
            state_name,
            year_l, month_dict[month_l],
            year_r, month_dict[month_r],
            fast_mode
        )

    with st.expander("🔍 Data Validation", expanded=True):
        st.write("### Panel Statistics")
        st.write(f"**Selected state:** {state_name}")
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
st.subheader(f"Comparative View — {state_name}")

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
        title=f"{gas}<br>{state_name}<br>{month_l} {year_l} ({unit})<br><span style='font-weight:400;'>Shared comparison scale</span>",
        vmin=meta["shared_min"],
        vmax=meta["shared_max"],
        colors=SHARED_PALETTE,
        position="bottomleft"
    )

    st_folium(
        left_map,
        height=550,
        width=500,
        key=f"left_map_{state_name}_{gas}_{month_l}_{year_l}_{opacity}",
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
        title=f"{gas}<br>{state_name}<br>{month_r} {year_r} ({unit})<br><span style='font-weight:400;'>Shared comparison scale</span>",
        vmin=meta["shared_min"],
        vmax=meta["shared_max"],
        colors=SHARED_PALETTE,
        position="bottomright"
    )

    st_folium(
        right_map,
        height=550,
        width=500,
        key=f"right_map_{state_name}_{gas}_{month_r}_{year_r}_{opacity}",
        returned_objects=[]
    )

# ---------------------------------------------------------
# DIFFERENCE MAP
# ---------------------------------------------------------
st.subheader(f"Difference Map for {state_name} (Right − Left)")

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
    title=f"{gas} Difference<br>{state_name}<br>({month_r} {year_r} − {month_l} {year_l}) ({unit})",
    vmin=meta["diff_min"],
    vmax=meta["diff_max"],
    colors=DIFF_PALETTE,
    position="bottomcenter"
)

st_folium(
    diff_map,
    height=500,
    width=1100,
    key=f"diff_map_{state_name}_{gas}_{month_l}_{year_l}_{month_r}_{year_r}_{opacity}",
    returned_objects=[]
)

# ---------------------------------------------------------
# DATA QUALITY NOTES
# ---------------------------------------------------------
with st.expander("📊 Data Quality Notes", expanded=False):
    st.markdown(f"""
**Selected region**
- `{state_name}`

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
