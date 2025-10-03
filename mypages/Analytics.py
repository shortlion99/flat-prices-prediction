import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import duckdb
import pydeck as pdk


# --- TOWN TO REGION MAPPING (TBC)---
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


def _norm_flat_label(s: str) -> str:
    return str(s).strip().upper().replace("-", " ").replace("/", " ").replace("  ", " ")


def render_altair(chart):
    try:
        return st.altair_chart(chart, width="stretch")
    except TypeError:
        return st.altair_chart(chart, use_container_width=True)


@st.cache_data(show_spinner=False)
def load_data(
    path: str = "data/hdb_df_geocoded_condensed.csv",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(path, low_memory=False)

    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype(str).str.strip()
    if "postal_sector" in df.columns:
        df["postal_sector"] = df["postal_sector"].astype(str).str.strip()

    if "town" in df.columns:
        region_series = df["town"].map(TOWN_TO_REGION)
        region_dummies = pd.get_dummies(region_series)
        if not region_dummies.empty:
            for c in region_dummies.columns:
                # convert to boolean dtype
                region_dummies[c] = region_dummies[c].astype(bool)
            df = pd.concat([df, region_dummies], axis=1)

    # date columns to datetime format conversion
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["year"] = df["month"].dt.year
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
    else:
        df["year"] = np.nan

    # one-hot encoding
    flat_cols = [col for col in df.columns if col.startswith("flat_type_")]
    if not flat_cols:
        source_col = next(
            (
                col
                for col in ["flat_type", "Flat_Type", "flat", "flat category"]
                if col in df.columns
            ),
            None,
        )
        if not source_col:
            raise ValueError("No valid flat-type column found.")
        df["_flat_norm"] = df[source_col].map(_norm_flat_label)
        dummies = pd.get_dummies(df["_flat_norm"], prefix="flat_type")
        dummies.columns = [col.replace(" ", "_") for col in dummies.columns]
        df = pd.concat([df.drop(columns=["_flat_norm"]), dummies], axis=1)
        flat_cols = list(dummies.columns)

    for c in flat_cols:
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        df[c] = (
            df[c].fillna(0).astype(int).astype(bool)
            if pd.api.types.is_numeric_dtype(df[c])
            else (
                df[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "t": True,
                        "f": False,
                        "yes": True,
                        "no": False,
                    }
                )
                .fillna(False)
                .astype(bool)
            )
        )

    flat_types = [col.replace("flat_type_", "").replace("_", " ") for col in flat_cols]

    # consistency
    rename_map = {
        "floor_area": "floor_area_sqm",
        "area_sqm": "floor_area_sqm",
        "remaining_lease_years": "remaining_lease",
        "mrt_km": "nearest_mrt_distance_km",
        "price": "resale_price",
        "ppsm": "price_per_sqm",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )

    return df, flat_cols, flat_types


@st.cache_resource(show_spinner=False)
def load_model(path: str = "models/best_xgboost_model.pkl"):
    """Load the trained model using joblib (important!)"""
    try:
        model = joblib.load(path)
        print(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align a single-row feature frame to model.feature_names_in_, mapping name variants.

    - If a required column is missing, try replacing spaces with underscores and vice versa.
    - Fill any still-missing columns with 0.
    - Return columns in the exact order expected by the model.
    """
    if not hasattr(model, "feature_names_in_"):
        return X

    aligned = {}
    for name in model.feature_names_in_:
        if name in X.columns:
            aligned[name] = X[name]
            continue
        candidate1 = name.replace(" ", "_")
        candidate2 = name.replace("_", " ")
        if candidate1 in X.columns:
            aligned[name] = X[candidate1]
        elif candidate2 in X.columns:
            aligned[name] = X[candidate2]
        else:
            aligned[name] = 0

    X_aligned = pd.DataFrame(aligned)
    for c in X_aligned.columns:
        if pd.api.types.is_bool_dtype(X_aligned[c]):
            X_aligned[c] = X_aligned[c].astype(float)
    return X_aligned


def render_price_banner(
    price: float | None, ci: tuple[float, float] | None, used_model: bool
):
    banner = st.container()
    with banner:
        if price is None:
            st.markdown(
                """
                <div class="price-banner">
                  <div style="font-size: 14px; color: #6b7280;">Predicted Price</div>
                  <div style="font-size: 28px; font-weight: 700; color: #374151;">Select filters to see an estimate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            subtitle = "Model prediction" if used_model else "Comps-based estimate"
            ci_html = (
                f'<div style="font-size:12px;color:#6b7280;">95% CI: SGD {ci[0]:,.0f} – {ci[1]:,.0f}</div>'
                if (not used_model and ci is not None)
                else ""
            )
            st.markdown(
                f"""
                <div class="price-banner price-banner--filled">
                  <div style="font-size: 13px; color: #6366f1;">{subtitle}</div>
                  <div style="font-size: 36px; font-weight: 800; color: #111827;">SGD {price:,.0f}</div>
                  {ci_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


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
        "price_per_sqm",
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

    for col in ["storey_bucket_Low", "storey_bucket_Mid"]:
        if col in row.index and col in user:
            row[col] = user[col]

    # df now has all columns the model expects
    return pd.DataFrame([row])


def comps_estimate(
    df: pd.DataFrame,
    town: str,
    flat_type: str,
    area: float,
    lease: float,
    mrt_km: float,
    area_tol=0.15,
    lease_tol=10,
):
    norm_ft = _norm_flat_label(flat_type)
    ft_col = "flat_type_" + norm_ft.replace(" ", "_")

    subset = df.copy()
    if "town" in subset.columns and town:
        subset = subset[subset["town"] == town]
    if ft_col in subset.columns:
        subset = subset[subset[ft_col]]
    if "floor_area_sqm" in subset.columns:
        subset = subset[
            subset["floor_area_sqm"].between(
                area * (1 - area_tol), area * (1 + area_tol)
            )
        ]
    if "remaining_lease" in subset.columns:
        subset = subset[
            subset["remaining_lease"].between(
                max(0, lease - lease_tol), min(99, lease + lease_tol)
            )
        ]
    if "nearest_mrt_distance_km" in subset.columns:
        subset = subset[subset["nearest_mrt_distance_km"] <= mrt_km]

    if subset.empty or "resale_price" not in subset.columns:
        return None, None

    prices = subset["resale_price"].dropna()
    point = float(prices.median())
    lo, hi = np.percentile(prices, [2.5, 97.5])

    trend = None
    if "year" in subset.columns:
        trend = (
            subset.groupby("year", as_index=False)["resale_price"]
            .median()
            .sort_values("year")
        )

    return (point, (lo, hi)), trend


def quick_forecast(trend_df: pd.DataFrame, years_ahead: int = 5):
    if trend_df is None or trend_df.empty or len(trend_df) < 2:
        return None
    x = trend_df["year"].values
    y = trend_df["resale_price"].values
    slope, intercept = np.polyfit(x, y, 1)
    future_years = np.arange(x.max() + 1, x.max() + 1 + years_ahead)
    y_pred = intercept + slope * future_years
    y_hat_hist = intercept + slope * x
    resid_std = (
        float(np.std(y - y_hat_hist))
        if len(y) > 2
        else (float(np.std(y)) if len(y) > 1 else 0.0)
    )
    return pd.DataFrame(
        {
            "year": future_years,
            "pred": y_pred,
            "lo": y_pred - 1.96 * resid_std,
            "hi": y_pred + 1.96 * resid_std,
        }
    )


# ---------------------------- Page ----------------------------
def show():
    st.title("🏡 HDB Analytics")
    st.caption(
        "Use predictive models and time-series forecasts to estimate property prices based on location and attributes."
    )
    st.markdown(
        """
        <style>
          :root { --card-bg: #ffffff; --muted: #6b7280; --border: #e5e7eb; --brand: #6366f1; }
          .kpi { border: 1px solid var(--border); border-radius: 12px; padding: 16px; background: var(--card-bg); text-align: center; }
          .kpi .label { font-size: 12px; color: var(--muted); }
          .kpi .value { font-size: 28px; font-weight: 800; margin-top: 6px; }
          .kpi .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
          .card { border: 1px solid var(--border); border-radius: 12px; padding: 20px; background: var(--card-bg); margin-bottom: 20px; min-height: 10px; display: flex; flex-direction: column; }
          .card h3 { margin: 0 0 0 0; font-size: 16px; font-weight: 600; }
          .card-content { flex: 1; }
          .section { margin-top: 12px; }
          /* Add spacing between KPI banner and first row */
          .kpi { margin-bottom: 30px; }
          /* Force consistent heights for charts and maps */
          .stAltair, .stAltair > div, .stAltair canvas, .stAltair svg { height: 300px !important; max-height: 300px !important; }
          .js-plotly-plot { height: 300px !important; max-height: 300px !important; }
          [data-testid="stPydeckChart"], [data-testid="stPydeckChart"] > div { height: 300px !important; max-height: 300px !important; }
          /* More aggressive targeting for card contents */
          .card .stAltair, .card .stAltair > div, .card .stAltair canvas, .card .stAltair svg { height: 300px !important; max-height: 300px !important; min-height: 300px !important; }
          .card [data-testid="stPydeckChart"], .card [data-testid="stPydeckChart"] > div { height: 300px !important; max-height: 300px !important; min-height: 300px !important; }
          /* Target the specific chart containers */
          .card-content .stAltair, .card-content .stAltair svg { height: 300px !important; max-height: 300px !important; }
          .card-content [data-testid="stPydeckChart"] { height: 300px !important; max-height: 300px !important; }
          /* Force SVG elements to specific height */
          .card svg.marks { height: 300px !important; max-height: 300px !important; }
          .stAltair svg { height: 300px !important; max-height: 300px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # data + model
    df, flat_cols, flat_types = load_data("data/hdb_df_geocoded_condensed.csv")
    model = load_model("models/best_xgboost_model.pkl")
    model_loaded = model is not None

    # sidebar filters
    with st.sidebar:
        st.header("Filters")
        town = st.selectbox(
            "Town",
            sorted(df["town"].dropna().unique().tolist())
            if "town" in df.columns
            else [],
        )
        flat_type = st.selectbox("Flat Type", flat_types, index=0)
        area = st.number_input(
            "Size (sqm)", min_value=20.0, max_value=200.0, value=80.0, step=5.0
        )
        lease = st.slider("Remaining Lease (years)", 0, 99, 60)
        st.markdown("---")
        mrt_km = st.slider("Max Distance to MRT (km)", 0.0, 5.0, 1.0, 0.1)
        schools_km = st.slider("Schools within (km)", 0.0, 5.0, 1.5, 0.1)
        childcare_km = st.slider("Childcare within (km)", 0.0, 5.0, 1.5, 0.1)
        supermarkets_km = st.slider("Supermarkets within (km)", 0.0, 5.0, 1.0, 0.1)
        hawker_km = st.slider("Hawker Centres within (km)", 0.0, 5.0, 1.0, 0.1)
        st.markdown("---")
        horizon = st.slider("Forecast Horizon (years)", 1, 10, 5)

    # build features
    user_payload = {
        "town": town,
        "flat_type": flat_type,
        "floor_area_sqm": float(area),
        "remaining_lease": float(lease),
        "nearest_mrt_distance_km": float(mrt_km),
        "nearest_schools_distance_km": float(schools_km),
        "nearest_childcare_distance_km": float(childcare_km),
        "nearest_supermarkets_distance_km": float(supermarkets_km),
        "nearest_hawker_distance_km": float(hawker_km),
    }
    X_user = build_feature_row(user_payload, flat_cols, df.columns)

    # compute prediction + comps
    model_price = None
    if model_loaded:
        try:
            X_for_model = _align_features_to_model(X_user, model)
            model_price = float(model.predict(X_for_model)[0])
        except Exception:
            model_price = None

    (est, ci), trend = (None, None), None
    if model_price is None:
        comp = comps_estimate(df, town, flat_type, area, lease, mrt_km)
        if comp is not None and comp[0] is not None:
            (est, ci), trend = comp

    price_to_show = model_price if model_price is not None else est

    # kpi header row
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        st.markdown(
            "<div class='kpi'><div class='label'>Predicted Price</div>"
            + (
                f"<div class='value'>SGD {price_to_show:,.0f}</div>"
                if price_to_show is not None
                else "<div class='value' style='color:#9CA3AF'>N/A</div>"
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='kpi'><div class='label'>Flat Type</div><div class='value'>"
            + flat_type
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi'><div class='label'>Area</div><div class='value'>{area:.0f} sqm</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='kpi'><div class='label'>Lease</div><div class='value'>{lease} yrs</div></div>",
            unsafe_allow_html=True,
        )

    # Prepare comparable data
    norm_ft = _norm_flat_label(flat_type)
    ft_col = "flat_type_" + norm_ft.replace(" ", "_")

    comp_df = df.copy()
    if "town" in comp_df.columns:
        comp_df = comp_df[comp_df["town"] == town]
    if ft_col in comp_df.columns:
        comp_df = comp_df[comp_df[ft_col]]
    if "floor_area_sqm" in comp_df.columns:
        comp_df = comp_df[comp_df["floor_area_sqm"].between(area * 0.85, area * 1.15)]
    if "remaining_lease" in comp_df.columns:
        comp_df = comp_df[
            comp_df["remaining_lease"].between(max(0, lease - 10), min(99, lease + 10))
        ]
    if "nearest_mrt_distance_km" in comp_df.columns:
        comp_df = comp_df[comp_df["nearest_mrt_distance_km"] <= mrt_km]

    # Prepare trend data
    if (trend is None or trend.empty) and ft_col in df.columns:
        broader = df[(df["town"] == town)] if "town" in df.columns else df
        if ft_col in broader.columns:
            broader = broader[broader[ft_col]]
        if "nearest_mrt_distance_km" in broader.columns:
            broader = broader[broader["nearest_mrt_distance_km"] <= mrt_km]
        if "resale_price" in broader.columns and "year" in broader.columns:
            trend = (
                broader.groupby("year", as_index=False)["resale_price"]
                .median()
                .sort_values("year")
            )

    # Comparable Price Distribution + Map
    col1_row1, col2_row1 = st.columns([2, 1])

    with col1_row1:
        st.markdown(
            "<div class='card'><h3>Comparable Price Distribution</h3><div class='card-content'>",
            unsafe_allow_html=True,
        )
        if not comp_df.empty and "resale_price" in comp_df.columns:
            hist = (
                alt.Chart(comp_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "resale_price:Q", bin=alt.Bin(maxbins=30), title="Resale Price"
                    ),
                    y=alt.Y("count():Q", title="Count"),
                )
                .properties(height=440)
            )
            render_altair(hist)
            st.caption(f"{len(comp_df)} comparable transactions")
        else:
            st.info("No comparable transactions under current filters.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2_row1:
        map_title = f"{town} Map" if town else "Select Town Map"
        st.markdown(
            f"<div class='card'><h3>{map_title}</h3><div class='card-content'>",
            unsafe_allow_html=True,
        )
        try:
            if town:
                con = duckdb.connect("data/hdb_df_geocoded_condensed.duckdb")
                query = (
                    "SELECT latitude, longitude, resale_price, nearest_mrt_name, nearest_schools_name "
                    "FROM resale WHERE town = ? AND latitude IS NOT NULL AND longitude IS NOT NULL LIMIT 10000"
                )
                map_df = con.execute(query, [town]).df()
                con.close()
                if not map_df.empty:
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["longitude", "latitude"],
                        get_radius=50,
                        get_fill_color=[149, 6, 6, 160],
                        pickable=True,
                    )
                    # center map roughly around selected town points
                    lat0 = float(map_df["latitude"].mean())
                    lon0 = float(map_df["longitude"].mean())
                    view_state = pdk.ViewState(
                        latitude=lat0, longitude=lon0, zoom=12, pitch=0
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                            initial_view_state=view_state,
                            layers=[layer],
                            tooltip={
                                "html": "<b>Price:</b> {resale_price}<br/><b>MRT:</b> {nearest_mrt_name}<br/><b>School:</b> {nearest_schools_name}",
                                "style": {"color": "white"},
                            },
                        )
                    )
                else:
                    st.info("No map points available for this town.")
            else:
                st.info("Select a town to view the map.")
        except Exception as e:
            st.info(f"Map unavailable: {e}")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Forecasted Price Trend + Trend Insights
    col1_row2, col2_row2 = st.columns([2, 1])

    with col1_row2:
        st.markdown(
            "<div class='card'><h3>Forecasted Price Trend</h3><div class='card-content'>",
            unsafe_allow_html=True,
        )
        if trend is not None and not trend.empty:
            trend = trend.dropna(subset=["year"]).copy()
            trend["year"] = trend["year"].astype(int)
            trend["date"] = pd.to_datetime(trend["year"], format="%Y")

            line_hist = (
                alt.Chart(trend)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Year", axis=alt.Axis(format="%Y")),
                    y=alt.Y("resale_price:Q", title="Median Price"),
                    tooltip=["year:O", "resale_price:Q"],
                )
            )

            fc = quick_forecast(trend, years_ahead=horizon)
            if fc is not None and not fc.empty:
                fc = fc.copy()
                fc["year"] = fc["year"].astype(int)
                fc["date"] = pd.to_datetime(fc["year"], format="%Y")

                band = (
                    alt.Chart(fc)
                    .mark_area(opacity=0.25)
                    .encode(
                        x=alt.X("date:T", title="Year", axis=alt.Axis(format="%Y")),
                        y=alt.Y("lo:Q", title="Price"),
                        y2="hi:Q",
                        tooltip=["year:O", "lo:Q", "hi:Q"],
                    )
                )
                line_pred = (
                    alt.Chart(fc)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T"),
                        y=alt.Y("pred:Q"),
                        tooltip=["year:O", "pred:Q"],
                    )
                )
                render_altair(band + line_hist + line_pred)
            else:
                render_altair(line_hist)
        else:
            st.info("Insufficient data for trend analysis.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2_row2:
        st.markdown(
            "<div class='card'><h3>Trend Insights</h3><div class='card-content'>",
            unsafe_allow_html=True,
        )
        if trend is not None and not trend.empty and len(trend) >= 3:
            min_year = trend.loc[trend["resale_price"].idxmin(), "year"]
            max_year = trend.loc[trend["resale_price"].idxmax(), "year"]
            min_price = trend["resale_price"].min()
            max_price = trend["resale_price"].max()
            recent_trend = trend.tail(3)["resale_price"].pct_change().mean() * 100

            st.markdown("**Historical Analysis:**")
            st.markdown(f"• **Lowest point:** {min_year} (SGD {min_price:,.0f})")
            st.markdown(f"• **Peak:** {max_year} (SGD {max_price:,.0f})")
            if not pd.isna(recent_trend):
                trend_direction = (
                    "rising"
                    if recent_trend > 0
                    else "falling"
                    if recent_trend < 0
                    else "stable"
                )
                st.markdown(
                    f"• **Recent trend:** {trend_direction} ({recent_trend:+.1f}% annually)"
                )

            # future forecast insights
            fc = quick_forecast(trend, years_ahead=horizon)
            if fc is not None and not fc.empty:
                future_price = fc["pred"].iloc[-1]
                current_price = trend["resale_price"].iloc[-1]
                price_change = ((future_price - current_price) / current_price) * 100
                st.markdown(
                    f"• **{horizon}-year outlook:** {price_change:+.1f}% change expected"
                )
        else:
            st.info("Insufficient data for trend insights.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Recent Comparable Sales + Feature Importance
    col1_row3, col2_row3 = st.columns([2, 1])

    with col1_row3:
        st.markdown(
            "<div class='card'><h3>Recent Comparable Sales</h3><div class='card-content'>",
            unsafe_allow_html=True,
        )
        if not comp_df.empty:
            cols_to_show = [
                c
                for c in [
                    "month",
                    "town",
                    "floor_area_sqm",
                    "remaining_lease",
                    "nearest_mrt_distance_km",
                    "resale_price",
                    "year",
                ]
                if c in comp_df.columns
            ]
            recent = comp_df.sort_values(
                ["year", "resale_price"], ascending=[False, True]
            )[cols_to_show].head(20)
            if not recent.empty:
                st.dataframe(recent, use_container_width=True)
            else:
                st.info("No recent comparable sales.")
        else:
            st.info("No comparable transactions under current filters.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2_row3:
        st.markdown(
            "<div class='card'><h3>Feature Importance </h3><div class='card-content'>",
            unsafe_allow_html=True,
        )
        if model_loaded and hasattr(model, "feature_importances_"):
            # use model's own feature names to avoid duplicates/misalignment
            if hasattr(model, "feature_names_in_"):
                names = pd.Index(model.feature_names_in_)
            else:
                names = pd.Index(X_user.columns)
            # build series and drop duplicate indices by keeping the first occurrence
            imp_series = pd.Series(model.feature_importances_, index=names)
            imp_series = imp_series[~imp_series.index.duplicated(keep="first")]
            imp = imp_series.sort_values(ascending=False).head(8)
            st.bar_chart(imp)
        else:
            st.write(
                "- Remaining Lease (↑) → higher price\n- Closer to MRT (↓ km) → higher price\n- Larger area (↑ sqm) → higher price\n- Town effects vary"
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.caption(
            "Data: `data/hdb_df_geocoded_condensed.csv`\n\n"
            + ("Model: loaded" if model_loaded else "Model: not loaded")
        )
