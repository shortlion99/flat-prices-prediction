import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import duckdb
import pydeck as pdk

# --- TOWN TO REGION MAPPING ---
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
    """Normalizes flat type string for consistent lookup."""
    return str(s).strip().upper().replace("-", " ").replace("/", " ").replace("  ", " ")


def render_altair(chart):
    """Renders Altair chart with container width."""
    return st.altair_chart(chart, use_container_width=True)


@st.cache_data(show_spinner="Loading data and preparing features...")
def load_data(path: str = "data/hdb_df_geocoded_condensed.duckdb"):
    """Loads and preprocesses HDB data from DuckDB."""
    con = duckdb.connect(path)
    df = con.execute(
        """
        SELECT 
            month, town, flat_type, resale_price, price_per_sqm,
            nearest_mrt_distance_km, nearest_supermarkets_distance_km,
            nearest_schools_distance_km, floor_area_sqm,
            remaining_lease, latitude, longitude
        FROM resale
        """
    ).df()
    con.close()

    # Create year column
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["year"] = df["month"].dt.year
    else:
        df["year"] = np.nan

    # Create Flat Type Dummies (essential for model and filtering)
    source_col = "flat_type" if "flat_type" in df.columns else None
    if source_col:
        df["_flat_norm"] = df[source_col].map(_norm_flat_label)
        dummies = pd.get_dummies(df["_flat_norm"], prefix="flat_type")
        dummies.columns = [col.replace(" ", "_") for col in dummies.columns]
        df = pd.concat([df.drop(columns=["_flat_norm"]), dummies], axis=1)

    flat_cols = [col for col in df.columns if col.startswith("flat_type_")]
    flat_types = [col.replace("flat_type_", "").replace("_", " ") for col in flat_cols]

    # Create Region Dummies (if town column exists)
    if "town" in df.columns:
        region_series = df["town"].map(TOWN_TO_REGION)
        region_dummies = pd.get_dummies(region_series)
        df = pd.concat([df, region_dummies], axis=1)

    return df, flat_cols, flat_types


@st.cache_resource(show_spinner="Loading machine learning model...")
def load_model(path: str = "models/best_rf_pipeline.pkl"):
    """Loads the fitted Scikit-learn pipeline model."""
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        return None


def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Aligns the user feature DataFrame to the features the model expects."""
    if not hasattr(model, "feature_names_in_"):
        return X

    # Creates a dictionary where keys are model features and values are the input values
    aligned = {
        name: X.get(name.replace(" ", "_"), 0.0) for name in model.feature_names_in_
    }
    X_aligned = pd.DataFrame(aligned)

    # Ensure all boolean/dummy columns are float for the model
    for c in X_aligned.columns:
        if pd.api.types.is_bool_dtype(X_aligned[c]):
            X_aligned[c] = X_aligned[c].astype(float)

    return X_aligned.iloc[[0]]


def build_feature_row(
    user: dict, flat_cols: list[str], df_columns: pd.Index
) -> pd.DataFrame:
    """Builds a single row DataFrame for prediction based on user inputs."""
    row = pd.Series(0.0, index=df_columns, dtype="float64")

    # Set continuous and proximity features
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

    # Set one-hot encoded features
    selected_flat_col = "flat_type_" + _norm_flat_label(user["flat_type"]).replace(
        " ", "_"
    )
    if selected_flat_col in row.index:
        row[selected_flat_col] = 1.0

    current_region_col = TOWN_TO_REGION.get(user.get("town"))
    if current_region_col and current_region_col in row.index:
        row[current_region_col] = 1.0

    return pd.DataFrame([row])


def quick_forecast(trend_df: pd.DataFrame, years_ahead: int = 5):
    """Performs a simple linear forecast based on historical median prices."""
    if trend_df is None or trend_df.empty or len(trend_df) < 2:
        return None

    x = trend_df["year"].values
    y = trend_df["resale_price"].values
    slope, intercept = np.polyfit(x, y, 1)
    future_years = np.arange(x.max() + 1, x.max() + 1 + years_ahead)
    y_pred = intercept + slope * future_years

    # Calculate standard deviation of residuals for confidence band
    resid_std = np.std(y - (intercept + slope * x))

    return pd.DataFrame(
        {
            "year": future_years,
            "pred": y_pred,
            "lo": y_pred - 1.96 * resid_std,
            "hi": y_pred + 1.96 * resid_std,
        }
    )


# ----------------------------------------------------------------------------------
# 🏡 HDB Analytics Dashboard Function
# ----------------------------------------------------------------------------------


def show():
    """Renders the main Streamlit dashboard page."""

    # --- Title and Caption ---
    st.title("🏡 HDB Analytics")
    st.caption(
        "Use predictive models and time-series forecasts to estimate property prices based on location and attributes."
    )

    # --- Custom CSS Styling (Enhanced for cleaner look) ---
    st.markdown(
        """ 
        <style> 
        /* General KPI styling for the clean look */
        :root { --card-bg: #ffffff; --muted: #6b7280; --border: #e5e7eb; --brand: #5B84B1FF; } 
        .kpi { border: 1px solid var(--border); border-radius: 12px; padding: 16px; background: var(--card-bg); text-align: center; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05); } 
        .kpi .label { font-size: 12px; color: var(--muted); margin-bottom: 4px; } 
        .kpi .value { font-size: 24px; font-weight: 700; color: #1f2937; } 
        
        /* Streamlit general tweaks */
        section.main { padding-top: 1rem; }
        .stContainer { margin-top: 1rem; }
        
        /* Custom horizontal rule for section separation */
        .st-emotion-cache-1pxe4x4 { /* Targets the default st.markdown("---") */
            margin-top: 1.5rem; 
            margin-bottom: 1.5rem; 
            border-top: 2px solid #ddd;
        }

        /* Hide the default text */ 
        [data-testid="stIconMaterial"] { 
        position: relative; display: inline-flex !important; align-items: center; justify-content: center; color: transparent !important; font-size: 0 !important; width: 24px; overflow: visible !important; } 
        
        /* Base: left arrow when open */ 
        [data-testid="stIconMaterial"]::before { content: "❮"; color: #6b7280 !important; font-family: system-ui, sans-serif; font-size: 20px !important; font-weight: 700; transition: transform 0.2s ease; } 
        [data-testid="stIconMaterial"][title="keyboard_double_arrow_right"]::before, [data-testid="stIconMaterial"][aria-label="keyboard_double_arrow_right"]::before, [data-testid="stIconMaterial"]:where([data-testid="stIconMaterial"]:not([title])):after { content: "❯"; }

        </style> 
        """,
        unsafe_allow_html=True,
    )

    # Load Data and Model
    df, flat_cols, flat_types = load_data("data/hdb_df_geocoded_condensed.duckdb")
    model = load_model("models/best_rf_pipeline.pkl")
    model_loaded = model is not None

    # ====================================================================
    # 📝 SIDEBAR FILTERS
    # ====================================================================
    with st.sidebar:
        st.header("🏠 Property Features ")

        # Core property features
        town = st.selectbox("Town", sorted(df["town"].dropna().unique().tolist()))
        flat_type = st.selectbox("Flat Type", flat_types, index=0)
        a_min = float(max(20.0, df["floor_area_sqm"].quantile(0.02)))
        a_max = float(min(200.0, df["floor_area_sqm"].quantile(0.98)))
        area = st.slider("Size (sqm)", a_min, a_max, float(np.clip(80.0, a_min, a_max)))
        lease = st.slider("Remaining Lease (years)", 0, 99, 60)

        # Model Proximity Features (for price prediction)
        st.subheader("🧠 Model Proximity Features")
        model_mrt_km = st.slider(
            "Nearest MRT Distance (km)", 0.0, 5.0, 1.0, 0.1, key="model_mrt"
        )
        model_schools_km = st.slider(
            "Nearest Schools Distance (km)", 0.0, 5.0, 1.5, 0.1, key="model_schools"
        )
        model_childcare_km = st.slider(
            "Nearest Childcare Distance (km)", 0.0, 5.0, 1.5, 0.1, key="model_childcare"
        )
        model_supermarkets_km = st.slider(
            "Nearest Supermarkets Distance (km)",
            0.0,
            5.0,
            1.0,
            0.1,
            key="model_supermarkets",
        )
        model_hawker_km = st.slider(
            "Nearest Hawker Centres Distance (km)",
            0.0,
            5.0,
            1.0,
            0.1,
            key="model_hawker",
        )

        st.markdown("---")

        st.header("📊 Historical Market Insights")
        # Comparable Filters (used only for historical chart/map filtering)
        comp_mrt_km = st.slider(
            "Max Distance to MRT (km)", 0.0, 5.0, 1.0, 0.1, key="comp_mrt"
        )
        comp_schools_km = st.slider(
            "Schools within (km)", 0.0, 5.0, 1.5, 0.1, key="comp_schools"
        )
        comp_supermarkets_km = st.slider(
            "Supermarkets within (km)", 0.0, 5.0, 1.0, 0.1, key="comp_supermarkets"
        )

        st.markdown("---")
        horizon = st.slider("Forecast Horizon (years)", 1, 10, 5)

    # --- Prepare Prediction Payload ---
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

    # --- Compute model-based prediction ---
    model_price = None
    if model_loaded:
        try:
            X_for_model = _align_features_to_model(X_user, model)
            price_per_sqm = float(model.predict(X_for_model)[0])
            model_price = price_per_sqm * user_payload["floor_area_sqm"]
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            model_price = None

    # ====================================================================
    st.header("🧠 Model-Based Insights")
    # ====================================================================
    with st.container(border=True):
        # Row 1: KPI header row
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            st.markdown(
                "<div class='kpi'><div class='label'>Predicted Price</div>"
                + (
                    f"<div class='value' style='color:#1f2937'>SGD {model_price:,.0f}</div>"
                    if model_price is not None
                    else "<div class='value' style='color:#9CA3AF'>N/A</div>"
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='kpi'><div class='label'>Flat Type</div><div class='value'>{flat_type}</div></div>",
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

        st.markdown(
            "<hr style='border:1px solid #e5e7eb; margin: 1.5rem 0 1rem 0;'>",
            unsafe_allow_html=True,
        )

        # Row 2 (Full Width): Feature Importance Chart
        st.subheader("Feature Importance")

        feature_importances = None
        names = X_user.columns

        if model_loaded:
            try:
                # Retrieve final estimator from pipeline
                if hasattr(model, "named_steps"):
                    final_estimator = list(model.named_steps.values())[-1]
                else:
                    final_estimator = model.steps[-1][1]

                if hasattr(final_estimator, "feature_importances_"):
                    feature_importances = final_estimator.feature_importances_

                names = getattr(model, "feature_names_in_", X_user.columns)

            except Exception as e:
                st.info(f"Could not extract model features. Error: {e}")

        if feature_importances is not None:
            imp_series = pd.Series(feature_importances, index=names)
            imp_series = imp_series[~imp_series.index.duplicated(keep="first")]
            imp_df = imp_series.sort_values(ascending=False).head(10).reset_index()
            imp_df.columns = ["Feature", "Importance"]

            # Altair Horizontal Bar Chart for maximum readability
            chart = (
                alt.Chart(imp_df)
                .mark_bar()
                .encode(
                    y=alt.Y("Feature:N", sort="-x", title="Feature Name"),
                    x=alt.X("Importance:Q", title="Relative Importance"),
                    color=alt.value("#5B84B1FF"),  # Solid brand color
                    tooltip=["Feature", "Importance"],
                )
                .properties(title="Top 10 Feature Importances", height=350)
            )
            render_altair(chart)
        else:
            st.info("Model feature importances unavailable. Check model configuration.")

        st.markdown(
            "<hr style='border:1px solid #e5e7eb; margin: 1rem 0 1.5rem 0;'>",
            unsafe_allow_html=True,
        )

        # Row 3 (Full Width Placeholder): Time-Series Forecast
        st.subheader("Time-Series Forecast (coming soon)")
        st.caption(
            "This section will show predicted price trends using an ARIMA/Prophet model trained on time-series data."
        )

    # ====================================================================
    st.header("📊 Historical Market Insights")
    # ====================================================================
    with st.container(border=True):
        # --- Data Filtering based on COMP Filters ---
        norm_ft = _norm_flat_label(flat_type)
        ft_col = "flat_type_" + norm_ft.replace(" ", "_")
        comp_df = df.copy()

        # Apply core filters
        comp_df = comp_df[comp_df["town"] == town]
        if ft_col in comp_df.columns:
            comp_df = comp_df[comp_df[ft_col]]
        comp_df = comp_df[comp_df["floor_area_sqm"].between(area * 0.85, area * 1.15)]
        comp_df = comp_df[
            comp_df["remaining_lease"].between(max(0, lease - 10), min(99, lease + 10))
        ]

        # Apply COMPARABLE FILTERS
        comp_df = comp_df[comp_df["nearest_mrt_distance_km"] <= comp_mrt_km]
        comp_df = comp_df[comp_df["nearest_schools_distance_km"] <= comp_schools_km]
        comp_df = comp_df[
            comp_df["nearest_supermarkets_distance_km"] <= comp_supermarkets_km
        ]

        # --- Trend Data (Uses Model Filters for trend scope) ---
        trend = None
        if ft_col in df.columns:
            broader = df[(df["town"] == town)]
            if ft_col in broader.columns:
                broader = broader[broader[ft_col]]
            if "resale_price" in broader.columns and "year" in broader.columns:
                trend = (
                    broader.groupby("year", as_index=False)["resale_price"]
                    .median()
                    .sort_values("year")
                )

        # Row 1: Comparable distribution + map
        col1_hist, col2_hist = st.columns([2, 1])
        with col1_hist:
            st.subheader("Comparable Price Distribution")
            if not comp_df.empty and "resale_price" in comp_df.columns:
                hist = (
                    alt.Chart(comp_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "resale_price:Q",
                            bin=alt.Bin(maxbins=30),
                            title="Resale Price (SGD)",
                        ),
                        y=alt.Y("count():Q", title="Count"),
                        tooltip=["resale_price:Q", "count():Q"],
                    )
                    .properties(height=450)  # Taller height for visibility
                )
                render_altair(hist)
                st.caption(f"Showing {len(comp_df)} comparable transactions.")
            else:
                st.info("No comparable transactions under current filters.")

        with col2_hist:
            st.subheader(f"{town} Map")
            try:
                if town and not comp_df.empty:
                    # Use a subset of comparables for faster map loading
                    map_df = (
                        comp_df[["latitude", "longitude", "resale_price"]]
                        .dropna()
                        .head(1000)
                    )

                    if not map_df.empty:
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=map_df,
                            get_position=["longitude", "latitude"],
                            get_radius=80,  # Slightly larger radius
                            get_fill_color=[170, 0, 0, 180],  # Darker, impactful color
                            pickable=True,
                        )
                        view_state = pdk.ViewState(
                            latitude=float(map_df["latitude"].mean()),
                            longitude=float(map_df["longitude"].mean()),
                            zoom=11.5,
                        )
                        st.pydeck_chart(
                            pdk.Deck(
                                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                                initial_view_state=view_state,
                                layers=[layer],
                                tooltip={
                                    "html": "<b>Price:</b> {resale_price:,.0f} SGD"
                                },
                            )
                        )
                    else:
                        st.info("No geocoded comparables found for map view.")
                else:
                    st.info("Select a town and filters to view the map.")
            except Exception as e:
                st.error(f"Map rendering error: {e}")

        st.markdown(
            "<hr style='border:1px solid #e5e7eb; margin: 1.5rem 0 1rem 0;'>",
            unsafe_allow_html=True,
        )

        # Row 2: Forecasted trend + insights
        col1_trend, col2_trend = st.columns([2, 1])
        with col1_trend:
            st.subheader("Forecasted Price Trend")
            if trend is not None and not trend.empty:
                trend = trend.dropna(subset=["year"]).copy()
                trend["year"] = trend["year"].astype(int)
                trend["date"] = pd.to_datetime(trend["year"], format="%Y")

                line_hist = (
                    alt.Chart(trend)
                    .mark_line(point=True, color="black")
                    .encode(
                        x=alt.X("date:T", title="Year", axis=alt.Axis(format="%Y")),
                        y=alt.Y("resale_price:Q", title="Median Price (SGD)"),
                        tooltip=[
                            "year:O",
                            alt.Tooltip("resale_price:Q", format=",.0f"),
                        ],
                    )
                )

                fc = quick_forecast(trend, years_ahead=horizon)
                if fc is not None and not fc.empty:
                    fc = fc.copy()
                    fc["year"] = fc["year"].astype(int)
                    fc["date"] = pd.to_datetime(fc["year"], format="%Y")

                    band = (
                        alt.Chart(fc)
                        .mark_area(opacity=0.3, color="#BCCBD9")
                        .encode(
                            x=alt.X("date:T"),
                            y=alt.Y("lo:Q", title="Price"),
                            y2="hi:Q",
                        )
                    )
                    line_pred = (
                        alt.Chart(fc)
                        .mark_line(
                            color="#A8201A", strokeDash=[5, 5]
                        )  # Dashed line for forecast
                        .encode(
                            x=alt.X("date:T"),
                            y=alt.Y("pred:Q"),
                            tooltip=["year:O", alt.Tooltip("pred:Q", format=",.0f")],
                        )
                    )
                    render_altair(band + line_hist + line_pred)
                else:
                    render_altair(line_hist)
            else:
                st.info("Insufficient data for trend analysis.")

        with col2_trend:
            st.subheader("Trend Insights")
            if trend is not None and not trend.empty and len(trend) >= 3:
                min_year = trend.loc[trend["resale_price"].idxmin(), "year"]
                max_year = trend.loc[trend["resale_price"].idxmax(), "year"]
                min_price = trend["resale_price"].min()
                max_price = trend["resale_price"].max()
                recent_trend = trend.tail(3)["resale_price"].pct_change().mean() * 100

                st.markdown("**Historical Analysis:**")
                st.markdown(f"- **Lowest point:** {min_year} (SGD {min_price:,.0f})")
                st.markdown(f"- **Peak:** {max_year} (SGD {max_price:,.0f})")
                if not pd.isna(recent_trend):
                    trend_direction = (
                        "rising"
                        if recent_trend > 0
                        else "falling"
                        if recent_trend < 0
                        else "stable"
                    )
                    st.markdown(
                        f"- **Recent trend:** **{trend_direction}** ({recent_trend:+.1f}% annually)"
                    )

                fc = quick_forecast(trend, years_ahead=horizon)
                if fc is not None and not fc.empty:
                    future_price = fc["pred"].iloc[-1]
                    current_price = trend["resale_price"].iloc[-1]
                    price_change = (
                        (future_price - current_price) / current_price
                    ) * 100
                    st.markdown(
                        f"- **{horizon}-year outlook:** **{price_change:+.1f}% change** expected"
                    )
            else:
                st.info("Insufficient data for trend insights.")

        st.markdown(
            "<hr style='border:1px solid #e5e7eb; margin: 1.5rem 0 1rem 0;'>",
            unsafe_allow_html=True,
        )

        # Row 3: Recent comparable sales table
        st.subheader("Recent Comparable Sales")
        if not comp_df.empty:
            cols_to_show = [
                "month",
                "town",
                "floor_area_sqm",
                "remaining_lease",
                "nearest_mrt_distance_km",
                "resale_price",
            ]
            recent = comp_df.sort_values("month", ascending=False)[cols_to_show].head(
                15
            )
            if not recent.empty:
                st.dataframe(
                    recent.style.format(
                        {"resale_price": "SGD {:,}", "month": "{:%Y-%m}"}
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No recent comparable sales.")
        else:
            st.info("No comparable transactions under current filters.")

    # Sidebar footer (model info)
    with st.sidebar:
        st.markdown("---")
        if model_loaded:
            try:
                _last_est = (
                    list(model.named_steps.values())[-1]
                    if hasattr(model, "named_steps")
                    else model.steps[-1][1]
                )
                _est_name = _last_est.__class__.__name__
            except Exception:
                _est_name = model.__class__.__name__

            _feat_cnt = len(getattr(model, "feature_names_in_", []))
            model_info = f"Model: `best_rf_pipeline.pkl` ({_est_name})" + (
                f" · features: {_feat_cnt}" if _feat_cnt is not None else ""
            )
        else:
            model_info = "Model: not loaded"

        st.caption(f"Data: `hdb_df_geocoded_condensed.duckdb`\n\n{model_info}")

        if model_loaded and _feat_cnt > 0:
            show_features = st.toggle("Show all model features", value=False)
            if show_features:
                st.dataframe(
                    pd.DataFrame({"Feature": list(model.feature_names_in_)}),
                    use_container_width=True,
                )


if __name__ == "__main__":
    show()
