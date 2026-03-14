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
    page_title="Atmospheric Gas Comparison (Sentinel-5P) for the United States",
    page_icon="🌎",
    layout="wide"
)

st.image("https://github.com/chrystali2002.png", width=100)
st.header("Timeframe Comparison of Atmospheric Gases over the United States", divider="rainbow")

st.write("""
This application enables users to visually compare atmospheric gas concentrations from Sentinel-5P over the United States
for two selected timeframes using a side-by-side interface, with a dynamic difference map showing changes (Right − Left).
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
# USA GEOMETRY
# ---------------------------------------------------------
usa = (
    ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    .filter(ee.Filter.eq("country_co", "US"))
    .geometry()
)

# ---------------------------------------------------------
# UI CONTROLS
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# PERFORMANCE TOGGLE
# ---------------------------------------------------------
with st.expander("⚡ Performance", expanded=False):
    fast_mode = st.toggle(
        "Fast mode (broad tile rendering + dynamic percentile legends)",
        value=True,
        help=(
            "Fast mode keeps map tiles responsive using broad preset ranges, "
            "while legends are still computed dynamically from Earth Engine percentiles."
        )
    )

# ---------------------------------------------------------
# OPACITY SLIDER
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# PRESET TILE VIS RANGES
# Used only for fast tile rendering.
# Legends are dynamic from percentile reduceRegion.
# ---------------------------------------------------------
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
        min-width: 220px;
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


def build_image(gas_key, year, month):
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
        .filterBounds(usa)
    )

    img = ee.Image(
        ee.Algorithms.If(
            col.size().gt(0),
            col.median(),
            ee.Image.constant(0).rename(band)
        )
    ).clip(usa).float()

    return img, stream, col_id


def get_percentile_range(img, band, fallback, p_low=2, p_high=98, scale=50000):
    try:
        stats = img.reduceRegion(
            reducer=ee.Reducer.percentile([p_low, p_high]),
            geometry=usa,
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


def get_percentile_diff_range(img_diff, band, fallback_abs, p_low=2, p_high=98, scale=50000):
    try:
        stats = img_diff.reduceRegion(
            reducer=ee.Reducer.percentile([p_low, p_high]),
            geometry=usa,
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
    year_l: int, month_l_num: int,
    year_r: int, month_r_num: int,
    fast_mode: bool
):
    band = gas_dict[gas_key]["band"]
    unit = gas_dict[gas_key]["unit"]

    img_left, stream_left, col_left = build_image(gas_key, year_l, month_l_num)
    img_right, stream_right, col_right = build_image(gas_key, year_r, month_r_num)
    img_diff = img_right.subtract(img_left).rename(band)

    fallback_min, fallback_max = FAST_RANGES.get(band, (0.0, 1.0))
    fallback_diff = max((fallback_max - fallback_min) * 0.3, 1e-6)

    # Dynamic legend ranges from Earth Engine percentiles
    left_leg_min, left_leg_max = get_percentile_range(
        img_left, band, (fallback_min, fallback_max),
        p_low=2, p_high=98,
        scale=80000 if fast_mode else 50000
    )
    right_leg_min, right_leg_max = get_percentile_range(
        img_right, band, (fallback_min, fallback_max),
        p_low=2, p_high=98,
        scale=80000 if fast_mode else 50000
    )
    diff_leg_min, diff_leg_max = get_percentile_diff_range(
        img_diff, band, fallback_diff,
        p_low=2, p_high=98,
        scale=80000 if fast_mode else 50000
    )

    # Tile rendering ranges
    if fast_mode:
        left_tile_min, left_tile_max = fallback_min, fallback_max
        right_tile_min, right_tile_max = fallback_min, fallback_max
        diff_tile_min, diff_tile_max = -fallback_diff, fallback_diff
    else:
        left_tile_min, left_tile_max = left_leg_min, left_leg_max
        right_tile_min, right_tile_max = right_leg_min, right_leg_max
        diff_tile_min, diff_tile_max = diff_leg_min, diff_leg_max

    viz_left = {
        "min": left_tile_min,
        "max": left_tile_max,
        "palette": SHARED_PALETTE,
    }
    viz_right = {
        "min": right_tile_min,
        "max": right_tile_max,
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

        # dynamic legend ranges
        "left_min": left_leg_min,
        "left_max": left_leg_max,
        "right_min": right_leg_min,
        "right_max": right_leg_max,
        "diff_min": diff_leg_min,
        "diff_max": diff_leg_max,

        # actual tile rendering ranges
        "left_tile_min": left_tile_min,
        "left_tile_max": left_tile_max,
        "right_tile_min": right_tile_min,
        "right_tile_max": right_tile_max,
        "diff_tile_min": diff_tile_min,
        "diff_tile_max": diff_tile_max,

        "stream_left": stream_left,
        "stream_right": stream_right,
        "col_left": col_left,
        "col_right": col_right,
    }

# ---------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------
try:
    with st.spinner("Preparing map layers and dynamic legends..."):
        meta = compute_tiles_and_meta(
            gas,
            year_l, month_dict[month_l],
            year_r, month_dict[month_r],
            fast_mode
        )

    with st.expander("🔍 Data Validation", expanded=True):
        st.write("### Panel Statistics")

        st.write(f"**Left panel ({month_l} {year_l})**")
        st.write(f"- Dynamic legend range: {meta['left_min']:.6f} to {meta['left_max']:.6f}")
        st.write(f"- Tile render range: {meta['left_tile_min']:.6f} to {meta['left_tile_max']:.6f}")

        st.write(f"**Right panel ({month_r} {year_r})**")
        st.write(f"- Dynamic legend range: {meta['right_min']:.6f} to {meta['right_max']:.6f}")
        st.write(f"- Tile render range: {meta['right_tile_min']:.6f} to {meta['right_tile_max']:.6f}")

        st.write("**Difference map**")
        st.write(f"- Dynamic legend range: {meta['diff_min']:.6f} to {meta['diff_max']:.6f}")
        st.write(f"- Tile render range: {meta['diff_tile_min']:.6f} to {meta['diff_tile_max']:.6f}")

        if meta["diff_min"] < 0 and meta["diff_max"] > 0:
            st.success("✅ Difference legend captures both increases and decreases")
        elif meta["diff_min"] >= 0:
            st.warning("⚠️ Difference legend suggests only increases")
        elif meta["diff_max"] <= 0:
            st.warning("⚠️ Difference legend suggests only decreases")

    # Optional pixel-level analysis
    if not fast_mode:
        with st.expander("📊 Pixel-by-Pixel Analysis", expanded=True):
            with st.spinner("Analyzing pixel-level variations..."):
                try:
                    unit = meta["unit"]
                    band = meta["band"]

                    img_left, _, _ = build_image(gas, year_l, month_dict[month_l])
                    img_right, _, _ = build_image(gas, year_r, month_dict[month_r])
                    img_diff = img_right.subtract(img_left)

                    img_left_renamed = img_left.rename(band + "_left")
                    img_right_renamed = img_right.rename(band + "_right")
                    img_diff_renamed = img_diff.rename(band + "_diff")
                    combined = img_left_renamed.addBands(img_right_renamed).addBands(img_diff_renamed)

                    samples = combined.sample(
                        region=usa,
                        scale=50000,
                        numPixels=1000,
                        seed=42,
                        geometries=True
                    )

                    sample_list = samples.getInfo()["features"]

                    left_values = []
                    right_values = []
                    diff_values = []
                    coords = []

                    for sample in sample_list:
                        props = sample["properties"]
                        geom = sample["geometry"]["coordinates"]

                        left_val = props.get(band + "_left")
                        right_val = props.get(band + "_right")
                        diff_val = props.get(band + "_diff")

                        if (
                            left_val is not None and right_val is not None and diff_val is not None
                            and not np.isnan(left_val)
                            and not np.isnan(right_val)
                            and not np.isnan(diff_val)
                        ):
                            left_values.append(left_val)
                            right_values.append(right_val)
                            diff_values.append(diff_val)
                            coords.append(geom)

                    left_values = np.array(left_values)
                    right_values = np.array(right_values)
                    diff_values = np.array(diff_values)

                    if len(diff_values) > 0:
                        st.write(f"### Analysis of {len(diff_values)} random pixels")

                        stat_col1, stat_col2, stat_col3 = st.columns(3)

                        with stat_col1:
                            st.write(f"**Left Panel ({month_l} {year_l})**")
                            st.write(f"Mean: {np.mean(left_values):.6f}")
                            st.write(f"StdDev: {np.std(left_values):.6f}")
                            st.write(f"Min: {np.min(left_values):.6f}")
                            st.write(f"Max: {np.max(left_values):.6f}")

                        with stat_col2:
                            st.write(f"**Right Panel ({month_r} {year_r})**")
                            st.write(f"Mean: {np.mean(right_values):.6f}")
                            st.write(f"StdDev: {np.std(right_values):.6f}")
                            st.write(f"Min: {np.min(right_values):.6f}")
                            st.write(f"Max: {np.max(right_values):.6f}")

                        with stat_col3:
                            st.write(f"**Difference ({month_r} {year_r} - {month_l} {year_l})**")
                            st.write(f"Mean: {np.mean(diff_values):.6f}")
                            st.write(f"StdDev: {np.std(diff_values):.6f}")
                            st.write(f"Min: {np.min(diff_values):.6f}")
                            st.write(f"Max: {np.max(diff_values):.6f}")

                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                        scatter = axes[0, 0].scatter(
                            left_values, right_values, alpha=0.5, s=10, c=diff_values, cmap="RdBu_r"
                        )
                        min_val = min(left_values.min(), right_values.min())
                        max_val = max(left_values.max(), right_values.max())
                        axes[0, 0].plot(
                            [min_val, max_val], [min_val, max_val],
                            "k--", label="1:1 line", alpha=0.5
                        )
                        axes[0, 0].set_xlabel(f"{month_l} {year_l} Values ({unit})")
                        axes[0, 0].set_ylabel(f"{month_r} {year_r} Values ({unit})")
                        axes[0, 0].set_title("Pixel-by-Pixel Comparison")
                        axes[0, 0].legend()
                        axes[0, 0].grid(True, alpha=0.3)
                        plt.colorbar(scatter, ax=axes[0, 0], label=f"Difference ({unit})")

                        axes[0, 1].hist(diff_values, bins=30, color="gray", edgecolor="black", alpha=0.7)
                        axes[0, 1].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
                        axes[0, 1].axvline(
                            x=np.mean(diff_values),
                            color="blue",
                            linestyle="-",
                            linewidth=2,
                            label=f"Mean: {np.mean(diff_values):.6f}"
                        )
                        axes[0, 1].set_xlabel(f"Difference Value ({unit})")
                        axes[0, 1].set_ylabel("Number of Pixels")
                        axes[0, 1].set_title("Distribution of Differences")
                        axes[0, 1].legend()
                        axes[0, 1].grid(True, alpha=0.3)

                        if coords and len(coords) == len(diff_values):
                            lons = [c[0] for c in coords]
                            lats = [c[1] for c in coords]
                            scatter = axes[1, 0].scatter(
                                lons, lats, c=diff_values, cmap="RdBu_r", s=30, alpha=0.7
                            )
                            axes[1, 0].set_xlabel("Longitude")
                            axes[1, 0].set_ylabel("Latitude")
                            axes[1, 0].set_title("Spatial Distribution of Differences")
                            plt.colorbar(scatter, ax=axes[1, 0], label=f"Difference ({unit})")
                        else:
                            axes[1, 0].text(
                                0.5, 0.5, "No coordinate data available",
                                horizontalalignment="center",
                                verticalalignment="center",
                                transform=axes[1, 0].transAxes
                            )

                        axes[1, 1].boxplot(
                            [left_values, right_values, diff_values],
                            tick_labels=[f"{year_l}", f"{year_r}", "Difference"]
                        )
                        axes[1, 1].set_ylabel(f"Value ({unit})")
                        axes[1, 1].set_title("Distribution Comparison")
                        axes[1, 1].grid(True, alpha=0.3)

                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No valid pixels found after filtering")

                except Exception as e:
                    st.warning(f"Could not analyze pixel variations: {str(e)}")
    else:
        with st.expander("📊 Pixel-by-Pixel Analysis", expanded=True):
            st.info("Fast mode keeps the app responsive. Turn it off for full pixel-level diagnostics.")

except Exception as e:
    st.error(f"Error preparing map layers: {str(e)}")
    st.stop()

band = meta["band"]
unit = meta["unit"]
stream_left = meta["stream_left"]
stream_right = meta["stream_right"]

# ---------------------------------------------------------
# SIDE-BY-SIDE MAPS
# ---------------------------------------------------------
st.subheader("Comparative View - Left vs Right")

label_col1, label_col2 = st.columns(2)
with label_col1:
    st.markdown(f"<h3 style='text-align: center;'>LEFT PANEL: {month_l} {year_l}</h3>", unsafe_allow_html=True)
with label_col2:
    st.markdown(f"<h3 style='text-align: center;'>RIGHT PANEL: {month_r} {year_r}</h3>", unsafe_allow_html=True)

map_col1, map_col2 = st.columns(2)

with map_col1:
    left_map = folium.Map(location=[40, -100], zoom_start=4, control_scale=True)

    folium.TileLayer(
        tiles=meta["left_tiles"],
        name=f"{month_l} {year_l}",
        attr="Sentinel-5P • Google Earth Engine",
        opacity=opacity
    ).add_to(left_map)

    folium.LayerControl().add_to(left_map)

    add_html_legend(
        left_map,
        title=f"{gas}<br>{month_l} {year_l} ({unit})",
        vmin=meta["left_min"],
        vmax=meta["left_max"],
        colors=SHARED_PALETTE,
        position="bottomleft"
    )

    st_folium(
        left_map,
        height=550,
        width=500,
        key=f"left_map_{gas}_{month_l}_{year_l}_{opacity}_{fast_mode}",
        returned_objects=[]
    )

with map_col2:
    right_map = folium.Map(location=[40, -100], zoom_start=4, control_scale=True)

    folium.TileLayer(
        tiles=meta["right_tiles"],
        name=f"{month_r} {year_r}",
        attr="Sentinel-5P • Google Earth Engine",
        opacity=opacity
    ).add_to(right_map)

    folium.LayerControl().add_to(right_map)

    add_html_legend(
        right_map,
        title=f"{gas}<br>{month_r} {year_r} ({unit})",
        vmin=meta["right_min"],
        vmax=meta["right_max"],
        colors=SHARED_PALETTE,
        position="bottomright"
    )

    st_folium(
        right_map,
        height=550,
        width=500,
        key=f"right_map_{gas}_{month_r}_{year_r}_{opacity}_{fast_mode}",
        returned_objects=[]
    )

# ---------------------------------------------------------
# DIFFERENCE MAP
# ---------------------------------------------------------
st.subheader("Difference Map (Right − Left)")

diff_map = folium.Map(location=[40, -100], zoom_start=4, control_scale=True)

folium.TileLayer(
    tiles=meta["diff_tiles"],
    name=f"Difference ({month_r} {year_r} - {month_l} {year_l})",
    attr="Sentinel-5P • Google Earth Engine",
    opacity=opacity
).add_to(diff_map)

folium.LayerControl().add_to(diff_map)

add_html_legend(
    diff_map,
    title=f"{gas} Difference<br>({month_r} {year_r} − {month_l} {year_l}) ({unit})",
    vmin=meta["diff_min"],
    vmax=meta["diff_max"],
    colors=DIFF_PALETTE,
    position="bottomcenter"
)

st_folium(
    diff_map,
    height=500,
    width=1100,
    key=f"diff_map_{gas}_{month_l}_{year_l}_{month_r}_{year_r}_{opacity}_{fast_mode}",
    returned_objects=[]
)

# ---------------------------------------------------------
# DATA QUALITY NOTES
# ---------------------------------------------------------
with st.expander("📊 Data Quality Notes", expanded=False):
    st.markdown(f"""
**Data Sources**
- Sentinel-5P Level 3 products from the Copernicus Programme
- OFFL (Offline) and NRTI (Near Real-Time) collections

**Current stream selection**
- Left panel stream: `{stream_left}`
- Right panel stream: `{stream_right}`

**Legend behavior**
- Legends are dynamic and recalculated from Earth Engine percentiles for each selected month/year
- In Fast mode, only tile rendering stays broad and fixed for speed
- In Precise mode, both tile rendering and legends are dynamic

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
