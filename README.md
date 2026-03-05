# 🎈

# 🛰️ ASOS Real-Time Quality Assurance & Control (ASOS-QA)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://asosqualityassurance.streamlit.app/)


## 📌 Overview

The **ASOS Quality Assurance** application provides a streamlined interface for the real-time acquisition, visualization, and rigorous Quality Control (QC) of surface weather observations from the **Automated Surface Observing System (ASOS)** network. 

While raw ASOS data is the backbone of U.S. aviation and forecasting, research-grade analysis requires the removal of "instrument noise" and sensor artifacts. This tool demonstartes how to ensure the data meets the high-fidelity standards required for climatological research, agricultural productivity modeling, and educational applications.

---

## ✨ Key Features

### ⚡ Real-Time Data Ingest
Directly interfaces with live data streams via the **Iowa Environmental Mesonet (IEM)** and **NOAA** to provide the most recent station observations without manual file handling.

### 🔍 Advanced QC Algorithms
Beyond simple range checks, the app applies sophisticated consistency tests to identify:
* **Temporal Spikes:** Filters unrealistic jumps in temperature or pressure that defy atmospheric physics.
* **Internal Inconsistencies:** Identifies logical errors, such as dew point temperatures exceeding ambient air temperatures.
* **Sensor Stagnation:** Detects "flatlined" data where a sensor may have become stuck or failed.



### 📊 Interactive Visualization
Dynamic plotting of key meteorological parameters, including:
* Temperature Extremes ($T_{max}$, $T_{min}$)
* Precipitation Accumulation


### 📥 Research-Ready Exports
Instantly download cleaned, QC-flagged datasets in formats compatible with modern climate modeling and GIS workflows (CSV/Excel).

---

## 🌾 Why Quality Assurance Matters

In agricultural and climate science, a single erroneous temperature spike can lead to a "false positive" for a heat stress event or a GDD (Growing Degree Day) miscalculation. By applying automated QC layers, this app:

1.  **Reduces Bias:** Removes instrument errors before they enter your statistical analysis.
2.  **Saves Time:** Automates the "data cleaning" phase, which typically consumes 80% of a researcher's lifecycle.
3.  **Ensures Reproducibility:** Provides a transparent, standardized method for handling missing data and outliers.

> **Technical Note:** This tool is specifically optimized for localized monitoring. It is a valuable resource for agricultural stakeholders, where high-accuracy weather data is critical for field-level projects and classroom science.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* [Streamlit](https://streamlit.io/)
* Pandas, Numpy, and Plotly

### Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/asos-quality-assurance.git](https://github.com/yourusername/asos-quality-assurance.git)

https://asosqualityassurance.streamlit.app/

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run ASOS_QA_QC_app.py
   ```
