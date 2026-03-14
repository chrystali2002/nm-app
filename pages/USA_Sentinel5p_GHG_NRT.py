#NEW
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
        "Fast mode (no dynamic stats; instant reruns)",
        value=True,
        help="Fast mode avoids reduceRegion/getInfo. Precise mode computes min/max using Earth Engine sampling (slower on first run, cached thereafter)."
    )

# ---------------------------------------------------------
# OPACITY SLIDER
# ---------------------------------------------------------
opacity = st.slider("Overlay Opacity", 0.1, 1.0, 0.8)

# ---------------------------------------------------------
# GAS CONFIG with Units
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
# FAST MODE VIS RANGES
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
# LEGEND HELPERS
# ---------------------------------------------------------
def fmt_tick(x):
    if x is None or np.isnan(x) or np.isinf(x):
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    if abs(x) >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2e}"


def add_html_legend(map_obj, title, vmin, vmax, colors, position="bottomleft"):
    """
    Add a custom HTML legend directly inside a Folium map.
    This avoids all legends sharing the same generic Leaflet CSS class.
    """
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
        min-width: 210px;
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


# ---------------------------------------------------------
# DATE HANDLING
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# EARTH ENGINE HELPERS
# ---------------------------------------------------------
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


def get_dynamic_range_simple(img, band):
    try:
        sample = img.sample(
            region=usa,
            scale=100000,
            numPixels=1000,
            seed=0,
            geometries=True
        )

        stats = sample.aggregate_stats(band)
        min_val = stats.get("min").getInfo()
        max_val = stats.get("max").getInfo()

        if min_val is None or max_val is None or np.isnan(min_val) or np.isnan(max_val):
            return FAST_RANGES.get(band, (0.0, 1.0))

        range_val = max_val - min_val
        min_val = min_val - 0.05 * range_val
        max_val = max_val + 0.05 * range_val

        return float(min_val), float(max_val)

    except Exception as e:
        st.warning(f"Could not compute dynamic range, using defaults: {str(e)}")
        return FAST_RANGES.get(band, (0.0, 1.0))

# ---------------------------------------------------------
# CACHED EE PIPELINE WITH RETRY LOGIC
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
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

    try:
        img_left, stream_left, col_left = build_image(gas_key, year_l, month_l_num)
        img_right, stream_right, col_right = build_image(gas_key, year_r, month_r_num)

        if fast_mode:
            left_min, left_max = FAST_RANGES.get(band, (0.0, 1.0))
            right_min, right_max = FAST_RANGES.get(band, (0.0, 1.0))
        else:
            left_min, left_max = get_dynamic_range_simple(img_left, band)
            right_min, right_max = get_dynamic_range_simple(img_right, band)

        img_diff = img_right.subtract(img_left).rename(band)

        if fast_mode:
            data_range = (left_max - left_min + right_max - right_min) / 2
            max_abs = data_range * 0.3
        else:
            try:
                sample = img_diff.sample(
                    region=usa,
                    scale=100000,
                    numPixels=1000,
                    seed=0
                )
                stats = sample.aggregate_stats(band)
                min_val = stats.get("min").getInfo()
                max_val = stats.get("max").getInfo()

                if min_val is not None and max_val is not None:
                    if not (np.isnan(min_val) or np.isnan(max_val)):
                        max_abs = max(abs(min_val), abs(max_val)) * 1.1
                    else:
                        data_range = (left_max - left_min + right_max - right_min) / 2
                        max_abs = data_range * 0.3
                else:
                    data_range = (left_max - left_min + right_max - right_min) / 2
                    max_abs = data_range * 0.3
            except Exception:
                data_range = (left_max - left_min + right_max - right_min) / 2
                max_abs = data_range * 0.3

        max_abs = max(max_abs, 1e-6)

        shared_min = min(left_min, right_min)
        shared_max = max(left_max, right_max)

        viz_left = {
            "min": left_min,
            "max": left_max,
            "palette": SHARED_PALETTE,
        }
        viz_right = {
            "min": right_min,
            "max": right_max,
            "palette": SHARED_PALETTE,
        }
        viz_diff = {
            "min": -max_abs,
            "max": max_abs,
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
            "left_min": left_min,
            "left_max": left_max,
            "right_min": right_min,
            "right_max": right_max,
            "shared_min": shared_min,
            "shared_max": shared_max,
            "diff_min": -max_abs,
            "diff_max": max_abs,
            "stream_left": stream_left,
            "stream_right": stream_right,
            "col_left": col_left,
            "col_right": col_right
        }

    except Exception as e:
        st.error(f"Error in compute_tiles_and_meta: {str(e)}")

        fallback_min, fallback_max = FAST_RANGES.get(band, (0.0, 1.0))
        fallback_diff_range = (fallback_max - fallback_min) * 0.3

        fallback_img = ee.Image.constant(0).rename(band).clip(usa)
        fallback_viz = {"min": 0, "max": 1, "palette": ["gray"]}
        fallback_tiles = fallback_img.getMapId(fallback_viz)["tile_fetcher"].url_format

        return {
            "band": band,
            "unit": unit,
            "left_tiles": fallback_tiles,
            "right_tiles": fallback_tiles,
            "diff_tiles": fallback_tiles,
            "left_min": fallback_min,
            "left_max": fallback_max,
            "right_min": fallback_min,
            "right_max": fallback_max,
            "shared_min": fallback_min,
            "shared_max": fallback_max,
            "diff_min": -fallback_diff_range,
            "diff_max": fallback_diff_range,
            "stream_left": "FALLBACK",
            "stream_right": "FALLBACK",
            "col_left": "FALLBACK",
            "col_right": "FALLBACK"
        }

# ---------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------
try:
    with st.spinner("Preparing map layers (this may take a moment)..."):
        meta = compute_tiles_and_meta(
            gas,
            year_l, month_dict[month_l],
            year_r, month_dict[month_r],
            fast_mode
        )

    # ===== DEBUG SECTION =====
    with st.expander("🔍 Data Validation", expanded=True):
        st.write("### Panel Statistics")
        st.write("Available keys in meta:", list(meta.keys()))

        if "left_min" in meta and "left_max" in meta:
            st.write(f"**Left Panel ({month_l} {year_l}):**")
            st.write(f"  - Overall Range: {meta['left_min']:.6f} to {meta['left_max']:.6f}")

        if "right_min" in meta and "right_max" in meta:
            st.write(f"**Right Panel ({month_r} {year_r}):**")
            st.write(f"  - Overall Range: {meta['right_min']:.6f} to {meta['right_max']:.6f}")

        if "diff_min" in meta and "diff_max" in meta:
            st.write(f"**Difference Map Range:** {meta['diff_min']:.6f} to {meta['diff_max']:.6f}")

            if meta["diff_min"] < 0 and meta["diff_max"] > 0:
                st.success("✅ Difference map shows BOTH increases and decreases")
            elif meta["diff_min"] >= 0:
                st.warning("⚠️ Difference map shows ONLY increases (all values >= 0)")
            elif meta["diff_max"] <= 0:
                st.warning("⚠️ Difference map shows ONLY decreases (all values <= 0)")

    # ===== PIXEL-BY-PIXEL ANALYSIS =====
    if not fast_mode:
        with st.expander("📊 Pixel-by-Pixel Analysis", expanded=True):
            with st.spinner("Analyzing pixel-by-pixel variations..."):
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

                    if len(sample_list) > 0:
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

                            n_increase = np.sum(diff_values > 0.0001)
                            n_decrease = np.sum(diff_values < -0.0001)
                            n_nochange = np.sum(np.abs(diff_values) <= 0.0001)

                            st.write("### Pixel-by-Pixel Changes")

                            col_inc, col_dec, col_nc = st.columns(3)
                            with col_inc:
                                st.markdown(
                                    f"<h3 style='color: red; text-align: center;'>{n_increase}</h3>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"<p style='text-align: center;'>INCREASES 🔴<br>({n_increase/len(diff_values)*100:.1f}%)</p>",
                                    unsafe_allow_html=True
                                )

                            with col_dec:
                                st.markdown(
                                    f"<h3 style='color: blue; text-align: center;'>{n_decrease}</h3>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"<p style='text-align: center;'>DECREASES 🔵<br>({n_decrease/len(diff_values)*100:.1f}%)</p>",
                                    unsafe_allow_html=True
                                )

                            with col_nc:
                                st.markdown(
                                    f"<h3 style='color: gray; text-align: center;'>{n_nochange}</h3>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"<p style='text-align: center;'>NO CHANGE ⚪<br>({n_nochange/len(diff_values)*100:.1f}%)</p>",
                                    unsafe_allow_html=True
                                )

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

                            box_data = [left_values, right_values, diff_values]
                            axes[1, 1].boxplot(
                                box_data,
                                tick_labels=[f"{year_l}", f"{year_r}", "Difference"]
                            )
                            axes[1, 1].set_ylabel(f"Value ({unit})")
                            axes[1, 1].set_title("Distribution Comparison")
                            axes[1, 1].grid(True, alpha=0.3)

                            plt.tight_layout()
                            st.pyplot(fig)

                            st.write("### Example Pixel Comparisons")
                            n_examples = min(5, len(diff_values))
                            if n_examples > 0:
                                example_indices = np.random.choice(len(diff_values), n_examples, replace=False)
                                example_data = []
                                for idx in example_indices:
                                    change_type = (
                                        "INCREASE 🔴" if diff_values[idx] > 0.0001
                                        else "DECREASE 🔵" if diff_values[idx] < -0.0001
                                        else "NO CHANGE ⚪"
                                    )
                                    example_data.append({
                                        f"{month_l} {year_l} Value": f"{left_values[idx]:.6f}",
                                        f"{month_r} {year_r} Value": f"{right_values[idx]:.6f}",
                                        "Difference": f"{diff_values[idx]:.6f}",
                                        "Change": change_type
                                    })

                                st.table(example_data)

                            st.write("### Conclusion")
                            mean_diff = np.mean(diff_values)
                            if mean_diff > 0.0001:
                                st.success(f"✅ **OVERALL INCREASE**: Mean difference = {mean_diff:.6f} {unit}")
                            elif mean_diff < -0.0001:
                                st.error(f"❌ **OVERALL DECREASE**: Mean difference = {mean_diff:.6f} {unit}")
                            else:
                                st.warning(f"⚠️ **NO OVERALL CHANGE**: Mean difference = {mean_diff:.6f} {unit}")

                            if n_increase > n_decrease:
                                st.info(
                                    f"More pixels show INCREASES ({n_increase}, {n_increase/len(diff_values)*100:.1f}%) "
                                    f"than DECREASES ({n_decrease}, {n_decrease/len(diff_values)*100:.1f}%)"
                                )
                            elif n_decrease > n_increase:
                                st.info(
                                    f"More pixels show DECREASES ({n_decrease}, {n_decrease/len(diff_values)*100:.1f}%) "
                                    f"than INCREASES ({n_increase}, {n_increase/len(diff_values)*100:.1f}%)"
                                )
                            else:
                                st.info(f"Equal number of increases and decreases ({n_increase} each)")
                        else:
                            st.warning("No valid pixels found after filtering")
                    else:
                        st.warning("No samples returned from Earth Engine")

                except Exception as e:
                    st.warning(f"Could not analyze pixel variations: {str(e)}")
                    st.exception(e)
    else:
        with st.expander("📊 Pixel-by-Pixel Analysis", expanded=True):
            st.info("Switch off Fast mode in the performance tab to see detailed pixel-by-pixel analysis")

except Exception as e:
    st.error(f"Error preparing map layers: {str(e)}")
    st.info("Try switching to Fast mode or selecting a different date range.")
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

    st_folium(left_map, height=550, width=500, key="left_map", returned_objects=[])

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

    st_folium(right_map, height=550, width=500, key="right_map", returned_objects=[])

# ---------------------------------------------------------
# DIFFERENCE MAP
# ---------------------------------------------------------
st.subheader("Difference Map (Right − Left)")

diff_map = folium.Map(location=[40, -100], zoom_start=4, control_scale=True)

diff_tile_name = f"Difference ({month_r} {year_r} - {month_l} {year_l})"

folium.TileLayer(
    tiles=meta["diff_tiles"],
    name=diff_tile_name,
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

st_folium(diff_map, height=500, width=1100, key="usa_diff_map", returned_objects=[])

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

**Important notes**
- Maps show monthly median concentrations
- Fast mode uses predefined visualization ranges
- Precise mode samples 1000 pixels for dynamic range estimation
- Black/blue indicates lower concentrations; red indicates higher concentrations

**Difference map**
- Shows absolute difference: **Right panel − Left panel**
- 🔴 **Red** = Increase (Right > Left)
- ⚪ **White** = No change
- 🔵 **Blue** = Decrease (Right < Left)

**Units**
- `mol/m²`: moles per square meter
- `ppbv`: parts per billion by volume
- `unitless`: no physical units

**Stream selection logic**
- NRTI: used for very recent dates
- OFFL: used for historical dates when available and more stable
""")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption("Data provided by Google Earth Engine and Copernicus Sentinel-5P")
