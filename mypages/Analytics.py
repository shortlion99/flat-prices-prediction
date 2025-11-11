import joblib
import numpy as np
import pandas as pd
import streamlit as st
from data.data_access import get_duckdb_conn
import altair as alt
import pydeck as pdk

# ============================================================
# Global Visual Theme (Altair + CSS)
# ============================================================

PALETTE = {
    "brand": "#4F46E5",  # indigo-600
    "brand_soft": "#A5B4FC",  # indigo-300
    "ink": "#0F172A",  # slate-900
    "muted": "#6B7280",  # gray-500
    "bg": "#F8FAFC",  # slate-50
    "card": "#FFFFFF",  # white
    "border": "#E5E7EB",  # gray-200
    "accent": "#06B6D4",  # cyan-500
    "accent_soft": "#CFFAFE",  # cyan-100
    "warn": "#F59E0B",  # amber-500
    "danger": "#EF4444",  # red-500
    "ok": "#10B981",  # emerald-500
}


def _set_altair_theme():
    alt.themes.register(
        "hdb_theme",
        lambda: {
            "config": {
                "font": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                "range": {
                    "category": [
                        PALETTE["brand"],
                        "#111827",
                        PALETTE["accent"],
                        "#9CA3AF",
                    ]
                },
                "axis": {
                    "labelColor": PALETTE["ink"],
                    "titleColor": PALETTE["ink"],
                    "gridColor": PALETTE["border"],
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                },
                "legend": {"labelColor": PALETTE["ink"], "titleColor": PALETTE["ink"]},
                "view": {"stroke": None},
            }
        },
    )
    alt.themes.enable("hdb_theme")


_set_altair_theme()

# ============================================================
# TOWN TO REGION
# ============================================================

TOWN_TO_REGION = {
    "ANG MO KIO": "region_Central",
    "BEDOK": "region_East",
    "BISHAN": "region_Central",
    "BUKIT BATOK": "region_West",
    "BUKIT MERAH": "region_Central",
    "BUKIT PANJANG": "region_West",
    "BUKIT TIMAH": "region_Central",
    "CENTRAL AREA": "region_Central",
    "CHOA CHU KANG": "region_West",
    "CLEMENTI": "region_West",
    "GEYLANG": "region_Central",
    "HOUGANG": "region_Northeast",
    "JURONG EAST": "region_West",
    "JURONG WEST": "region_West",
    "KALLANG/WHAMPOA": "region_Central",
    "MARINE PARADE": "region_East",
    "PASIR RIS": "region_East",
    "PUNGGOL": "region_Northeast",
    "QUEENSTOWN": "region_Central",
    "SEMBAWANG": "region_North",
    "SENGKANG": "region_Northeast",
    "SERANGOON": "region_Northeast",
    "TAMPINES": "region_East",
    "TOA PAYOH": "region_Central",
    "WOODLANDS": "region_North",
    "YISHUN": "region_North",
}

# ============================================================
# Helpers
# ============================================================


def _norm_flat_label(s: str) -> str:
    return str(s).strip().upper().replace("-", " ").replace("/", " ").replace("  ", " ")


def fmt_money(x):
    try:
        return f"SGD {x:,.0f}"
    except Exception:
        return "SGD —"


def render_altair(chart):
    return st.altair_chart(chart, use_container_width=True)


def _kpi(value, label, help_text=None, color=PALETTE["ink"]):
    st.markdown(
        f"""
        <div class='kpi'>
            <div class='label'>{label}</div>
            <div class='value' style='color:{color}'>{value}</div>
            {f"<div class='hint'>{help_text}</div>" if help_text else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Data + Models
# ============================================================


@st.cache_data(show_spinner="Loading data and preparing features…")
def load_data():
    con = get_duckdb_conn()
    df = con.execute(
        """
        SELECT 
            month, town, flat_type, resale_price, price_per_sqm,
            nearest_mrt_distance_km, 
            nearest_supermarkets_distance_km, 
            nearest_schools_distance_km, 
            nearest_childcare_distance_km, 
            nearest_hawker_distance_km,    
            floor_area_sqm,
            remaining_lease, latitude, longitude,
            district_number
        FROM resale
        """
    ).df()

    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.normalize()
        df["year"] = df["month"].dt.year
    else:
        df["year"] = np.nan

    if "flat_type" in df.columns:
        df["_flat_norm"] = df["flat_type"].map(_norm_flat_label)
        dummies = pd.get_dummies(df["_flat_norm"], prefix="flat_type")
        dummies.columns = [c.replace(" ", "_") for c in dummies.columns]
        df = pd.concat([df.drop(columns=["_flat_norm"]), dummies], axis=1)

    flat_cols = [c for c in df.columns if c.startswith("flat_type_")]
    flat_types = [c.replace("flat_type_", "").replace("_", " ") for c in flat_cols]

    if "town" in df.columns:
        region_series = df["town"].map(TOWN_TO_REGION)
        region_dummies = pd.get_dummies(region_series)
        df = pd.concat([df, region_dummies], axis=1)

    return df, flat_cols, flat_types


@st.cache_resource(show_spinner="Loading machine learning model…")
def load_model(path: str = "models/best_random_forest_model.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading time-series model (SARIMAX)…")
def load_sarimax_model(path: str = "models/sarimax_flattype_district.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return None


def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    if not hasattr(model, "feature_names_in_"):
        return X
    aligned = {
        name: X.get(name.replace(" ", "_"), 0.0) for name in model.feature_names_in_
    }
    X_aligned = pd.DataFrame(aligned)
    for c in X_aligned.columns:
        if pd.api.types.is_bool_dtype(X_aligned[c]):
            X_aligned[c] = X_aligned[c].astype(float)
    return X_aligned.iloc[[0]]


def build_feature_row(
    user: dict, flat_cols: list[str], df_columns: pd.Index
) -> pd.DataFrame:
    row = pd.Series(0.0, index=df_columns, dtype="float64")
    for feature in [
        "floor_area_sqm",
        "remaining_lease",
        "nearest_schools_distance_km",
        "nearest_childcare_distance_km",
        "nearest_supermarkets_distance_km",
        "nearest_hawker_distance_km",
        "nearest_mrt_distance_km",
    ]:
        if feature in row.index and feature in user:
            row[feature] = user[feature]

    selected_flat_col = "flat_type_" + _norm_flat_label(user["flat_type"]).replace(
        " ", "_"
    )
    if selected_flat_col in row.index:
        row[selected_flat_col] = 1.0

    current_region_col = TOWN_TO_REGION.get(user.get("town"))
    if current_region_col and current_region_col in row.index:
        row[current_region_col] = 1.0

    return pd.DataFrame([row])


# ============================================================
# SARIMAX Forecast Helpers
# ============================================================


def _make_forecast_index_like_model(model, steps: int):
    try:
        row_labels = getattr(model.model.data, "row_labels", None)
    except Exception:
        row_labels = None

    if row_labels is None:
        nobs = getattr(model.model, "nobs", None) or getattr(model, "nobs", None) or 0
        return pd.RangeIndex(start=int(nobs), stop=int(nobs) + steps)

    if isinstance(row_labels, pd.DatetimeIndex):
        last = row_labels[-1]
        freq = row_labels.freq or "MS"
        start = last + pd.tseries.frequencies.to_offset(freq)
        return pd.date_range(start=start, periods=steps, freq=freq)

    if isinstance(row_labels, pd.PeriodIndex):
        return pd.period_range(
            start=row_labels[-1] + 1, periods=steps, freq=row_labels.freq
        )

    try:
        nobs = len(row_labels)
        return pd.RangeIndex(start=nobs, stop=nobs + steps)
    except Exception:
        nobs = getattr(model.model, "nobs", None) or 0
        return pd.RangeIndex(start=int(nobs), stop=int(nobs) + steps)


def prepare_and_run_forecast(
    sarimax_model, df: pd.DataFrame, user_payload: dict, forecast_months: int = 12
):
    if sarimax_model is None:
        return None

    df_work = df.copy()
    if "month" not in df_work.columns:
        raise ValueError("'month' column not found in dataset")
    if not pd.api.types.is_datetime64_any_dtype(df_work["month"]):
        df_work["month"] = pd.to_datetime(df_work["month"], errors="coerce")

    valid_months = df_work["month"].dropna()
    if valid_months.empty:
        raise ValueError("No valid month data in dataset")

    start_date_forecast = valid_months.max() + pd.DateOffset(months=1)
    display_future_dates = pd.date_range(
        start=start_date_forecast, periods=forecast_months, freq="MS"
    )

    selected_town = user_payload.get("town")
    norm_flat_type = _norm_flat_label(user_payload.get("flat_type"))

    town_mask = df_work["town"] == selected_town
    if town_mask.any() and "district_number" in df_work.columns:
        district_series = df_work.loc[town_mask, "district_number"].dropna()
        district_num = (
            float(district_series.mode().iloc[0]) if not district_series.empty else 0.0
        )
    else:
        district_num = 0.0

    if hasattr(sarimax_model, "model") and hasattr(sarimax_model.model, "exog_names"):
        exog_features = sarimax_model.model.exog_names
    elif hasattr(sarimax_model, "exog_names"):
        exog_features = sarimax_model.exog_names
    elif hasattr(sarimax_model, "model") and hasattr(sarimax_model.model, "exog"):
        if sarimax_model.model.exog is not None and hasattr(
            sarimax_model.model.exog, "columns"
        ):
            exog_features = sarimax_model.model.exog.columns.tolist()
        else:
            raise ValueError("Cannot determine exogenous variable names from model")
    else:
        raise ValueError("Cannot determine exogenous variable names from model")

    compat_index = _make_forecast_index_like_model(sarimax_model, forecast_months)
    exog_forecast_df = pd.DataFrame(0.0, index=compat_index, columns=exog_features)

    if "district_number" in exog_features:
        exog_forecast_df["district_number"] = district_num

    ft_variations = [
        "flat_type_" + norm_flat_type.replace(" ", "_"),
        "flat_type_" + norm_flat_type,
        "flat_type_" + norm_flat_type.replace("_", " "),
    ]
    matching_col = None
    for ft_col in ft_variations:
        if ft_col in exog_features:
            matching_col = ft_col
            break
    if matching_col is None:
        available_ft_cols = [f for f in exog_features if "flat_type" in f.lower()]
        if available_ft_cols:
            norm_ft_clean = (
                norm_flat_type.upper()
                .replace(" ", "")
                .replace("_", "")
                .replace("-", "")
            )
            for col in available_ft_cols:
                col_ft_part = (
                    col.replace("flat_type_", "")
                    .upper()
                    .replace(" ", "")
                    .replace("_", "")
                    .replace("-", "")
                )
                if norm_ft_clean == col_ft_part:
                    matching_col = col
                    break

    if matching_col:
        exog_forecast_df[matching_col] = 1.0
    else:
        available_ft_cols = [f for f in exog_features if "flat_type" in f.lower()]
        raise ValueError(
            f"Could not match flat type '{norm_flat_type}' to any exog feature. "
            f"Tried: {ft_variations}. Available: {available_ft_cols}"
        )

    if exog_forecast_df.isna().any().any():
        bad = exog_forecast_df.columns[exog_forecast_df.isna().any()].tolist()
        raise ValueError(f"Exogenous forecast DataFrame contains NaN values in {bad}")

    try:
        forecast_results = sarimax_model.get_forecast(
            steps=forecast_months, exog=exog_forecast_df.values
        )
    except Exception:
        forecast_results = sarimax_model.get_forecast(
            steps=forecast_months, exog=exog_forecast_df
        )

    if forecast_results is None:
        raise ValueError("Forecast returned None")

    forecast_df = forecast_results.summary_frame(alpha=0.05)
    if forecast_df is None or forecast_df.empty:
        raise ValueError("Forecast summary_frame is empty.")

    if "mean" not in forecast_df.columns:
        for alt_col in ["predicted_mean", "forecast", "yhat"]:
            if alt_col in forecast_df.columns:
                forecast_df = forecast_df.rename(columns={alt_col: "mean"})
                break
    if "mean" not in forecast_df.columns:
        raise ValueError(f"Forecast missing 'mean' column: {list(forecast_df.columns)}")

    if "mean_ci_lower" not in forecast_df.columns:
        if "lower" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"lower": "mean_ci_lower"})
        else:
            raise ValueError("Forecast missing 'mean_ci_lower'.")

    if "mean_ci_upper" not in forecast_df.columns:
        if "upper" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"upper": "mean_ci_upper"})
        else:
            raise ValueError("Forecast missing 'mean_ci_upper'.")

    forecast_df["month"] = display_future_dates

    # Level calibration to last observed median ppsqm
    tmp = df_work.copy()
    tmp.loc[tmp["floor_area_sqm"] == 0, "floor_area_sqm"] = np.nan
    tmp["ppsqm"] = tmp["resale_price"] / tmp["floor_area_sqm"]
    last_month = pd.to_datetime(tmp["month"], errors="coerce").max()
    last_hist_ppsqm = float(tmp.loc[tmp["month"] == last_month, "ppsqm"].median())

    try:
        one_step = sarimax_model.get_forecast(
            steps=1, exog=exog_forecast_df.values[:1]
        ).summary_frame(alpha=0.05)
        model_next_ppsqm = float(one_step["mean"].iloc[0])
    except Exception:
        one_step = sarimax_model.get_forecast(
            steps=1, exog=exog_forecast_df.iloc[:1]
        ).summary_frame(alpha=0.05)
        model_next_ppsqm = float(one_step["mean"].iloc[0])

    cal = last_hist_ppsqm / model_next_ppsqm if model_next_ppsqm > 0 else 1.0
    if np.isfinite(cal) and cal > 0:
        forecast_df["mean"] *= cal
        forecast_df["mean_ci_lower"] *= cal
        forecast_df["mean_ci_upper"] *= cal

    area = float(user_payload.get("floor_area_sqm", 80.0))
    forecast_df["predicted_mean"] = forecast_df["mean"] * area
    forecast_df["mean_ci_lower"] = forecast_df["mean_ci_lower"] * area
    forecast_df["mean_ci_upper"] = forecast_df["mean_ci_upper"] * area

    numeric_cols = ["predicted_mean", "mean_ci_lower", "mean_ci_upper"]
    forecast_df[numeric_cols] = forecast_df[numeric_cols].clip(lower=0)

    forecast_df = forecast_df[["month"] + numeric_cols]
    if forecast_df.empty:
        raise ValueError("Final forecast DataFrame is empty after processing")

    return forecast_df


# ============================================================
# UI: Page
# ============================================================


def _inject_css():
    st.markdown(
        """
    <style>
    /* ===============================
       Global: fonts & color tokens
       =============================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded');

    :root{
      --brand:#4F46E5;        /* indigo-600 */
      --brand-soft:#A5B4FC;   /* indigo-300 */
      --ink:#0F172A;          /* slate-900 */
      --muted:#6B7280;        /* gray-500 */
      --card:#FFFFFF;         /* white */
      --border:#E5E7EB;       /* gray-200 */
      --accent:#06B6D4;       /* cyan-500 */
      --accent-soft:#CFFAFE;  /* cyan-100 */
    }

    /* ===============================
       Backgrounds → pure white
       =============================== */
    html, body,
    [data-testid="stAppViewContainer"]{
      background:#FFFFFF !important;
    }

    /* Use the Material Symbols ligature font for Streamlit's icon nodes */
    [data-testid="stSidebar"] [data-testid="stIconMaterial"],
    header [data-testid="stIconMaterial"]{
    font-family: 'Material Symbols Rounded' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    font-size: 22px !important;
    color: #6B7280 !important;
    line-height: 1;
    }

    /* Optional: cursor + hover for the collapse button area */
    [data-testid="stSidebarCollapseControl"],
    [data-testid="stSidebarNavCollapse"],
    [data-testid="stSidebarNavSeparator"]{
    cursor: pointer !important;
    }

 

    /* ===============================
       Typography
       =============================== */
    html, body, *{
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif !important;
    }
    .main-title{
      font-weight:800; letter-spacing:-0.02em; 
      background:linear-gradient(90deg, var(--brand), var(--accent));
      -webkit-background-clip:text; 
    }
    .subcap{ color:var(--muted); margin-top:-6px; }

    /* ===============================
       Cards / KPIs
       =============================== */
    .kpi{
      background:var(--card);
      border:1px solid var(--border);
      border-radius:14px;
      padding:14px;
      text-align:center;
      box-shadow:0 1px 2px rgba(2,6,23,0.03);
      min-height:94px;              /* keep KPI tiles visually equal */
      display:flex; flex-direction:column; justify-content:center;
    }
    .kpi .label{ font-size:12px; color:var(--muted); margin-bottom:2px; }
    .kpi .value{ font-weight:800; font-size:22px; color:var(--ink); }
    .kpi .hint { font-size:11px; color:var(--muted); margin-top:6px; }

    /* Soft badge (used for horizon) */
    .soft-badge{
      display:inline-block; padding:2px 8px; border-radius:999px;
      background:var(--accent-soft); color:var(--accent); font-size:11px;
      border:1px solid #A5F3FC;
    }

    /* Dividers, notes, footnotes */
    .divider{ height:1px; background:var(--border); margin:16px 0 8px 0; }
    .sidebar-note{
      font-size:12px; color:var(--muted); border-left:3px solid var(--brand);
      padding-left:8px; margin-top:8px;
    }
    .footnote{ color:var(--muted); font-size:12px; margin-top:6px; }

    /* Optional: neutralize default card strokes in main area */
    .st-emotion-cache-ocqkz7, .st-emotion-cache-1r6slb0{ border-color:var(--border) !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def show():
    _inject_css()

    st.markdown("<h1 class='main-title'>HDB Analytics</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subcap'>Estimate prices, explore market trends, and preview the next few months.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Data + models
    df, flat_cols, flat_types = load_data()
    rf_model = load_model("models/best_random_forest_model.pkl")
    model = rf_model
    model_loaded = model is not None
    sarimax_model = load_sarimax_model("models/sarimax_flattype_district.pkl")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("### Filters")
        town = st.selectbox("Town", sorted(df["town"].dropna().unique().tolist()))
        flat_type = st.selectbox("Flat Type", flat_types, index=0)

        # --- size slider limits based on selected town + flat type ---
        # Build the dummy-column name for the chosen flat type
        ft_norm = _norm_flat_label(flat_type)  # e.g. "2 ROOM"
        ft_col = "flat_type_" + ft_norm.replace(" ", "_")  # e.g. "flat_type_2_ROOM"

        # Filter to selected town + flat type (if the dummy exists)
        mask = df["town"] == town
        if ft_col in df.columns:
            mask &= df[ft_col] == 1

        sizes = df.loc[mask, "floor_area_sqm"].dropna()

        # Robust bounds (clip extreme outliers); fallback to global if empty
        if not sizes.empty:
            low = float(np.floor(sizes.quantile(0.05)))
            high = float(np.ceil(sizes.quantile(0.95)))
            default = float(np.median(sizes))
        else:
            # fallback to overall distribution
            all_sizes = df["floor_area_sqm"].dropna()
            low = (
                float(np.floor(all_sizes.quantile(0.05)))
                if not all_sizes.empty
                else 20.0
            )
            high = (
                float(np.ceil(all_sizes.quantile(0.95)))
                if not all_sizes.empty
                else 200.0
            )
            default = 80.0

        # Safety: ensure min < max and default inside range
        if high <= low:
            high = low + 1.0
        default = float(np.clip(default, low, high))

        area = st.slider(
            "Size (sqm)",
            low,
            high,
            default,
            step=1.0,
            help="Range is derived from recent sales for this town and flat type.",
        )

        lease = st.slider("Remaining Lease (years)", 0, 99, 60)

        forecast_horizon = st.slider(
            "Forecast Horizon (months)",
            3,
            36,
            12,
            1,
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Proximity (Predicted Price)")
        model_mrt_km = st.slider("Nearest MRT Distance (km)", 0.0, 5.0, 1.0, 0.1)
        model_schools_km = st.slider(
            "Nearest Schools Distance (km)", 0.0, 5.0, 1.5, 0.1
        )
        model_childcare_km = st.slider(
            "Nearest Childcare Distance (km)", 0.0, 5.0, 1.5, 0.1
        )
        model_supermarkets_km = st.slider(
            "Nearest Supermarkets Distance (km)", 0.0, 5.0, 1.0, 0.1
        )
        model_hawker_km = st.slider(
            "Nearest Hawker Centres Distance (km)", 0.0, 5.0, 1.0, 0.1
        )

    # Build prediction payload
    user_payload = {
        "town": town,
        "flat_type": flat_type,
        "floor_area_sqm": float(area),
        "remaining_lease": float(lease),
        "nearest_mrt_distance_km": float(model_mrt_km),
        "nearest_schools_distance_km": float(model_schools_km),
        "nearest_childcare_distance_km": float(model_childcare_km),
        "nearest_supermarkets_distance_km": float(model_supermarkets_km),
        "nearest_hawker_distance_km": float(model_hawker_km),
    }
    X_user = build_feature_row(user_payload, flat_cols, df.columns)

    # Predict
    model_price = None
    if model_loaded:
        try:
            X_for_model = _align_features_to_model(X_user, model)
            price_per_sqm = float(model.predict(X_for_model)[0])
            model_price = price_per_sqm * user_payload["floor_area_sqm"]
        except Exception as e:
            st.warning(f"Prediction error: {e}")

    # Shared datasets
    norm_ft = _norm_flat_label(flat_type)
    ft_col = "flat_type_" + norm_ft.replace(" ", "_")
    comp_df = df.copy()
    comp_df = comp_df[comp_df["town"] == town]
    if ft_col in comp_df.columns:
        comp_df = comp_df[comp_df[ft_col]]
    comp_df = comp_df[comp_df["floor_area_sqm"].between(area * 0.85, area * 1.15)]
    comp_df = comp_df[
        comp_df["remaining_lease"].between(max(0, lease - 10), min(99, lease + 10))
    ]
    price_dist_df = df[df["town"] == town].copy()

    trend_monthly = None
    forecast_df = None
    forecast_error = None

    if ft_col in df.columns:
        broader = df[(df["town"] == town)]
        if ft_col in broader.columns:
            broader = broader[broader[ft_col]]

        required_cols = {"resale_price", "floor_area_sqm", "month"}
        if required_cols.issubset(broader.columns):
            broader_month_clean = broader.copy()
            if not pd.api.types.is_datetime64_any_dtype(broader_month_clean["month"]):
                broader_month_clean["month"] = pd.to_datetime(
                    broader_month_clean["month"], errors="coerce"
                )
            broader_month_clean = broader_month_clean[
                broader_month_clean["month"].notna()
            ]

            ppsqm_df = broader_month_clean.copy()
            ppsqm_df.loc[ppsqm_df["floor_area_sqm"] == 0, "floor_area_sqm"] = np.nan
            ppsqm_df["price_per_sqm_obs"] = (
                ppsqm_df["resale_price"] / ppsqm_df["floor_area_sqm"]
            )

            hist_ppsqm = (
                ppsqm_df.dropna(subset=["price_per_sqm_obs"])
                .groupby("month", as_index=False)["price_per_sqm_obs"]
                .median()
                .rename(columns={"price_per_sqm_obs": "price_per_sqm"})
                .sort_values("month")
            )
            hist_ppsqm["month"] = pd.to_datetime(
                hist_ppsqm["month"], errors="coerce"
            ).dt.normalize()

            user_area = float(user_payload.get("floor_area_sqm", 80.0))
            trend_monthly = hist_ppsqm.copy()
            trend_monthly["price"] = trend_monthly["price_per_sqm"] * user_area
            trend_monthly["type"] = "Historical (area-adjusted)"

            if sarimax_model is not None:
                try:
                    forecast_df = prepare_and_run_forecast(
                        sarimax_model,
                        broader_month_clean,
                        user_payload,
                        forecast_months=forecast_horizon,
                    )
                    if forecast_df is not None and forecast_df.empty:
                        forecast_error = "Forecast returned empty results"
                        forecast_df = None
                except Exception as e:
                    forecast_error = str(e)
                    forecast_df = None

    # ---------------- Header KPIs (equal width) ----------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])  # equal widths
    with c1:
        _kpi(
            fmt_money(model_price) if model_price is not None else "—",
            "Predicted Price",
            help_text=f"{town} · {flat_type}",
        )
    with c2:
        _kpi(
            f"{area:.0f} sqm",
            "Size",
            help_text="Floor area",
        )
    with c3:
        _kpi(f"{lease} yrs", "Remaining Lease", help_text="User-selected")
    with c4:
        horizon_badge = f"<span class='soft-badge'>{forecast_horizon} months</span>"
        _kpi(horizon_badge, "Forecast Horizon", help_text="SARIMAX window")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ---------------- Tabs: Forecast · Market · Comparables ----------------
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🧭 Market", "📋 Comparables"])

    # Forecast Tab
    with tab1:
        st.markdown("#### Median Price Forecast")
        st.caption("Area-adjusted history + forecast with 95% confidence band.")

        if (
            trend_monthly is not None
            and not trend_monthly.empty
            and forecast_df is not None
            and not forecast_df.empty
        ):
            last_hist_date = trend_monthly["month"].max()
            start_date_hist_limit = last_hist_date - pd.DateOffset(months=24)
            filtered_hist_df = trend_monthly[
                trend_monthly["month"] >= start_date_hist_limit
            ].copy()

            forecast_plot_df = forecast_df.copy().rename(
                columns={"predicted_mean": "price"}
            )
            forecast_plot_df["type"] = "SARIMAX Forecast"
            forecast_plot_df["ci_lower"] = forecast_plot_df["mean_ci_lower"]
            forecast_plot_df["ci_upper"] = forecast_plot_df["mean_ci_upper"]

            all_vals = pd.concat(
                [filtered_hist_df[["price"]], forecast_plot_df[["price"]]],
                ignore_index=True,
            )["price"].dropna()
            if not all_vals.empty:
                y_min = float(all_vals.min())
                y_max = float(all_vals.max())
                pad = 0.06 * (y_max - y_min if y_max > y_min else max(1.0, y_max))
                y_domain = [max(0, y_min - pad), y_max + pad]
            else:
                y_domain = [0, 1]

            ci_band = (
                alt.Chart(forecast_plot_df)
                .mark_area(opacity=0.18, color=PALETTE["brand"])
                .encode(
                    x=alt.X("month:T", title=None),
                    y="mean_ci_lower:Q",
                    y2="mean_ci_upper:Q",
                )
            )

            gap_row = pd.DataFrame(
                {
                    "month": [last_hist_date + pd.DateOffset(days=1)],
                    "price": [None],
                    "type": ["Gap"],
                }
            )
            combined_df = pd.concat(
                [
                    filtered_hist_df[["month", "price"]].assign(type="Historical"),
                    gap_row[["month", "price", "type"]],
                    forecast_plot_df[
                        ["month", "price", "type", "ci_lower", "ci_upper"]
                    ],
                ],
                ignore_index=True,
            )

            for d in (combined_df, forecast_plot_df):
                d["month"] = pd.to_datetime(d["month"], errors="coerce")
                d["price"] = pd.to_numeric(d["price"], errors="coerce")

            hover = alt.selection_single(
                fields=["month"],
                nearest=True,
                on="mouseover",
                empty="none",
                clear="mouseout",
            )

            line = (
                alt.Chart(combined_df)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X(
                        "month:T",
                        title="Month",
                        scale=alt.Scale(
                            domain=[start_date_hist_limit, forecast_df["month"].max()]
                        ),
                        axis=alt.Axis(format="%b %Y", labelAngle=0, labelOverlap=True),
                    ),
                    y=alt.Y(
                        "price:Q",
                        title="Median Resale Price (SGD)",
                        scale=alt.Scale(domain=y_domain, zero=False),
                        axis=alt.Axis(format=",.0f", tickCount=6, grid=True),
                    ),
                    color=alt.condition(
                        alt.datum.type == "SARIMAX Forecast",
                        alt.value(PALETTE["brand"]),
                        alt.value("#111827"),
                    ),
                    detail="type:N",
                )
            )

            points = (
                line.mark_point(size=35, filled=True)
                .transform_filter(hover)
                .encode(color=alt.value("#111827"))
            )
            rule = (
                alt.Chart(combined_df)
                .mark_rule(strokeDash=[4, 4])
                .encode(x="month:T")
                .transform_filter(hover)
            )
            tooltip = (
                alt.Chart(combined_df)
                .mark_rule(opacity=0)
                .encode(
                    x="month:T",
                    y="price:Q",
                    tooltip=[
                        alt.Tooltip("month:T", title="Date", format="%b %Y"),
                        alt.Tooltip("price:Q", title="Median Price", format=",.0f"),
                        alt.Tooltip(
                            "ci_lower:Q", title="95% CI (lower)", format=",.0f"
                        ),
                        alt.Tooltip(
                            "ci_upper:Q", title="95% CI (upper)", format=",.0f"
                        ),
                        alt.Tooltip("type:N", title="Series"),
                    ],
                )
                .add_selection(hover)
            )

            final_chart = (
                (ci_band + line + points + rule + tooltip)
                .properties(
                    title=alt.TitleParams(
                        f"{forecast_horizon}-Month Median Price Forecast · {town} · {flat_type}",
                        anchor="start",
                        fontSize=15,
                        color=PALETTE["ink"],
                    ),
                    padding={"left": 60, "right": 16, "top": 10, "bottom": 30},
                    height=360,
                )
                .interactive(bind_y=False)
            )

            render_altair(final_chart)

            latest_forecast = forecast_df["predicted_mean"].iloc[-1]
            last_hist_total = trend_monthly["price"].iloc[-1]
            pct_change = (
                ((latest_forecast - last_hist_total) / last_hist_total * 100)
                if last_hist_total
                else np.nan
            )
            st.success(
                f"Projected **{forecast_horizon}-month median price**: **{fmt_money(latest_forecast)}** "
                f"({pct_change:+.1f}% vs last observed)."
            )

            col_dl1, col_dl2 = st.columns([1, 1])
            with col_dl1:
                st.download_button(
                    "⬇️ Download Forecast CSV",
                    data=forecast_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"forecast_{town}_{flat_type.replace(' ', '_').lower()}_{forecast_horizon}m.csv",
                    mime="text/csv",
                )
            with col_dl2:
                hist_slice = filtered_hist_df[["month", "price"]].rename(
                    columns={"price": "historical_price"}
                )
                st.download_button(
                    "⬇️ Download Last 24M History CSV",
                    data=hist_slice.to_csv(index=False).encode("utf-8"),
                    file_name=f"history_{town}_{flat_type.replace(' ', '_').lower()}_24m.csv",
                    mime="text/csv",
                )

        else:
            if sarimax_model is None:
                st.warning(
                    "ARIMAX model not loaded. Ensure `models/sarimax_flattype_district.pkl` exists."
                )
            elif trend_monthly is None or trend_monthly.empty:
                st.info(
                    "Insufficient historical data for the selected flat type and town."
                )
            elif forecast_df is None or forecast_df.empty:
                if forecast_error:
                    st.error(
                        "Forecast failed:\n\n"
                        f"- {forecast_error}\n\n"
                        "Check exogenous features and town/flat compatibility."
                    )
                else:
                    st.warning("Forecast unavailable for this combination.")
            else:
                st.info("No forecast available.")

    # Market Tab
    with tab2:
        st.markdown("#### Town-wide Distribution")
        st.caption("All resale transactions in this town, regardless of size or type.")

        if not price_dist_df.empty and "resale_price" in price_dist_df.columns:
            hist = (
                alt.Chart(price_dist_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "resale_price:Q",
                        bin=alt.Bin(maxbins=35),
                        title="Resale Price (SGD)",
                    ),
                    y=alt.Y("count():Q", title="Transactions"),
                    tooltip=[alt.Tooltip("count():Q", title="Transactions")],
                )
                .properties(height=420)
            )
            render_altair(hist)
            st.caption(
                f"Showing **{len(price_dist_df):,}** transactions in **{town}**."
            )
        else:
            st.info("No price data available for this town.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Map")
        st.caption(
            "Switch to view either all town transactions or just close comparables."
        )
        map_toggle = st.toggle("Show comparables only", value=False)

        try:
            current_map_df = comp_df if map_toggle else price_dist_df
            map_view_label = "Comparable Sales" if map_toggle else "Town Transactions"

            if town and not current_map_df.empty:
                st.caption(f"Displaying **{map_view_label}** on the map.")

                map_df = (
                    current_map_df.dropna(subset=["latitude", "longitude"])
                    .head(1000)
                    .copy()
                )
                if not map_df.empty:
                    map_df["resale_price_k"] = (map_df["resale_price"] / 1000).round(0)
                    map_df["price_per_sqm"] = map_df["price_per_sqm"].round(0)
                    map_df["floor_area_sqm"] = map_df["floor_area_sqm"].round(0)
                    map_df["month_str"] = pd.to_datetime(
                        map_df["month"], errors="coerce"
                    ).dt.strftime("%b %Y")
                    map_df["lease_str"] = (
                        map_df["remaining_lease"].round(0).astype(str) + " yrs"
                    )

                    for col in [
                        "nearest_mrt_distance_km",
                        "nearest_schools_distance_km",
                        "nearest_childcare_distance_km",
                        "nearest_supermarkets_distance_km",
                        "nearest_hawker_distance_km",
                    ]:
                        if col in map_df.columns:
                            map_df[col] = map_df[col].round(2)

                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["longitude", "latitude"],
                        get_radius=80,
                        get_fill_color=[79, 70, 229, 180],  # brand indigo
                        pickable=True,
                    )
                    view_state = pdk.ViewState(
                        latitude=float(map_df["latitude"].mean()),
                        longitude=float(map_df["longitude"].mean()),
                        zoom=11.5,
                    )
                    map_tooltip = {
                        "html": """
                            <div style="font-size: 13px; padding: 6px; color: #111;">
                                <b>📅 Month:</b> {month_str} <br/>
                                <b>💰 Price:</b> ${resale_price_k}k <br/>
                                <b>📐 Price/sqm:</b> ${price_per_sqm} <br/>
                                <b>🏠 Flat:</b> {flat_type}, {floor_area_sqm} sqm <br/>
                                <b>⏳ Lease:</b> {lease_str} <br/>
                                <b>🚇 MRT:</b> {nearest_mrt_distance_km} km <br/>
                                <b>🏫 School:</b> {nearest_schools_distance_km} km <br/>
                                <b>👶 Childcare:</b> {nearest_childcare_distance_km} km <br/>
                                <b>🛒 Supermarket:</b> {nearest_supermarkets_distance_km} km <br/>
                                <b>🍜 Hawker:</b> {nearest_hawker_distance_km} km
                            </div>
                        """,
                        "style": {
                            "backgroundColor": "rgba(255,255,255,0.95)",
                            "color": "black",
                            "border-radius": "6px",
                        },
                    }
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                            initial_view_state=view_state,
                            layers=[layer],
                            tooltip=map_tooltip,
                        )
                    )
                else:
                    st.info("No geocoded points to plot under current filters.")
            else:
                st.info("Select a town and ensure there are transactions to display.")
        except Exception as e:
            st.error(f"Map rendering error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Comparables Tab
    with tab3:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Recent Closely Comparable Sales")
        st.caption("Matches your size, type, lease and town filters.")

        if not comp_df.empty:
            cols_to_show = [
                "month",
                "floor_area_sqm",
                "remaining_lease",
                "nearest_mrt_distance_km",
                "nearest_schools_distance_km",
                "nearest_childcare_distance_km",
                "nearest_supermarkets_distance_km",
                "nearest_hawker_distance_km",
                "resale_price",
            ]
            for c in cols_to_show:
                if c not in comp_df.columns:
                    comp_df[c] = np.nan

            recent = comp_df.sort_values("month", ascending=False)[cols_to_show].head(
                25
            )
            if not recent.empty:
                fmt = {
                    "month": "{:%Y-%m}",
                    "floor_area_sqm": "{:.1f}",
                    "remaining_lease": "{:.0f}",
                    "nearest_mrt_distance_km": "{:.2f} km",
                    "nearest_schools_distance_km": "{:.2f} km",
                    "nearest_childcare_distance_km": "{:.2f} km",
                    "nearest_supermarkets_distance_km": "{:.2f} km",
                    "nearest_hawker_distance_km": "{:.2f} km",
                    "resale_price": "SGD {:,.2f}",
                }
                st.dataframe(recent.style.format(fmt), use_container_width=True)

                st.download_button(
                    "⬇️ Download Comparables CSV",
                    data=recent.to_csv(index=False).encode("utf-8"),
                    file_name=f"comparables_{town}_{flat_type.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No recent comparable sales matching all filters.")
        else:
            st.info("No comparable transactions under current filters.")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    show()
