import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


def _norm_flat_label(s: str) -> str:
    """Normalize a flat-type label so one-hots are consistent."""
    return str(s).strip().upper().replace("-", " ").replace("/", " ").replace("  ", " ")


@st.cache_data(show_spinner=False)
def load_data(
    path: str = "data/hdb_df_geocoded_condensed.csv",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(path)

    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["year"] = df["month"].dt.year
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
    elif "year" not in df.columns:
        df["year"] = np.nan

    # if flat-type one-hots already exist, ensure underscores in column names
    flat_cols = [c for c in df.columns if c.startswith("flat_type_")]
    if flat_cols:
        rename_underscored = {c: c.replace(" ", "_") for c in flat_cols if " " in c}
        if rename_underscored:
            df.rename(columns=rename_underscored, inplace=True)
        flat_cols = [c for c in df.columns if c.startswith("flat_type_")]

    # else, build them from a categorical column
    if not flat_cols:
        source_col = next(
            (
                c
                for c in ["flat_type", "Flat_Type", "flat", "flat category"]
                if c in df.columns
            ),
            None,
        )
        if source_col is None:
            raise ValueError(
                "No 'flat_type_*' one-hots or categorical flat type column found."
            )
        df["_flat_norm"] = df[source_col].map(_norm_flat_label)  # e.g., "3 ROOM"
        dummies = pd.get_dummies(
            df["_flat_norm"], prefix="flat_type"
        )  # e.g., "flat_type_3 ROOM"
        dummies.columns = [
            c.replace(" ", "_") for c in dummies.columns
        ]  # -> "flat_type_3_ROOM"
        df = pd.concat([df.drop(columns=["_flat_norm"]), dummies], axis=1)
        flat_cols = list(dummies.columns)

    # handling of 1/0 and "True"/"False"
    for c in flat_cols:
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int).astype(bool)
        else:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "t": True,
                        "1": True,
                        "yes": True,
                        "false": False,
                        "f": False,
                        "0": False,
                        "no": False,
                    }
                )
                .fillna(False)
                .astype(bool)
            )

    flat_types = [c.replace("flat_type_", "").replace("_", " ") for c in flat_cols]

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
def load_model(path: str = "model.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def build_feature_row(
    user: dict, flat_cols: list[str], df_columns: pd.Index
) -> pd.DataFrame:
    """Create a one-row DataFrame aligned to the training schema."""
    row = pd.Series(0.0, index=df_columns, dtype="float64")

    for k in [
        "floor_area_sqm",
        "remaining_lease",
        "nearest_schools_distance_km",
        "nearest_childcare_distance_km",
        "nearest_supermarkets_distance_km",
        "nearest_hawker_distance_km",
        "nearest_mrt_distance_km",
        "price_per_sqm",
    ]:
        if k in row.index and k in user:
            row[k] = user[k]

    for col in flat_cols:
        if col in row.index:
            row[col] = 0.0
    sel_flat_col = "flat_type_" + _norm_flat_label(user["flat_type"]).replace(" ", "_")
    if sel_flat_col in row.index:
        row[sel_flat_col] = 1.0

    for col in ["storey_bucket_Low", "storey_bucket_Mid"]:
        if col in row.index and col in user:
            row[col] = user[col]

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
    """Comparable sales estimate + median-by-year trend."""
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
    # ≤ max MRT distance
    if "nearest_mrt_distance_km" in subset.columns:
        subset = subset[subset["nearest_mrt_distance_km"] <= mrt_km]

    if subset.empty or "resale_price" not in subset.columns:
        return None, None

    prices = subset["resale_price"].dropna()
    if prices.empty:
        return None, None

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
    """Tiny linear projection with residual-based band."""
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
    st.title("Analytics — Preference Explorer")

    # Load data + (optional) model
    df, flat_cols, flat_types = load_data("data/hdb_df_geocoded_condensed.csv")
    model = load_model("model.pkl")
    model_loaded = model is not None
    st.caption(
        "Data loaded from **data/hdb_df_geocoded_condensed.csv**"
        + (
            " · Model loaded"
            if model_loaded
            else " · No model yet — using data-driven estimates"
        )
    )

    col_left, col_mid, col_right = st.columns([1, 1.2, 1])

    with col_left:
        st.subheader("User Inputs")
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
        mrt_km = st.slider("Max Distance to MRT (km)", 0.0, 5.0, 1.0, 0.1)
        horizon = st.slider("Forecast Horizon (years)", 1, 10, 5)

        with st.expander("More Filters"):
            schools_km = st.slider("Schools within (km)", 0.0, 5.0, 1.5, 0.1)
            childcare_km = st.slider("Childcare within (km)", 0.0, 5.0, 1.5, 0.1)
            supermarkets_km = st.slider("Supermarkets within (km)", 0.0, 5.0, 1.0, 0.1)
            hawker_km = st.slider("Hawker Centres within (km)", 0.0, 5.0, 1.0, 0.1)

        go = st.button("Generate Insights", use_container_width=True)

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

    if go:
        st.subheader("Results")

        left, right = st.columns([1.2, 1])

        with left:
            model_price = None
            if model_loaded:
                try:
                    model_price = float(model.predict(X_user)[0])
                except Exception as e:
                    st.info(
                        f"Model loaded but could not predict with current features. Falling back to data estimate. ({e})"
                    )

            (est, ci), trend = (None, None), None
            if model_price is None:
                comp = comps_estimate(df, town, flat_type, area, lease, mrt_km)
                if comp is None or comp[0] is None:
                    st.warning(
                        "No close comparables found with the current filters. Try widening your criteria."
                    )
                else:
                    (est, ci), trend = comp

            price_to_show = model_price if model_price is not None else est
            if price_to_show is not None:
                st.metric("Current Predicted Price", f"SGD {price_to_show:,.0f}")
                if model_price is None and ci is not None:
                    st.caption(
                        f"95% CI: SGD {ci[0]:,.0f} – {ci[1]:,.0f} (from comparable sales)"
                    )

        with right:
            st.markdown("**Feature Importance / Sensitivity**")
            if model_loaded and hasattr(model, "feature_importances_"):
                imp = (
                    pd.Series(model.feature_importances_, index=X_user.columns)
                    .sort_values(ascending=False)
                    .head(8)
                )
                st.bar_chart(imp)
            else:
                st.write(
                    "- Remaining Lease (↑) → higher price\n"
                    "- Closer to MRT (↓ km) → higher price\n"
                    "- Larger area (↑ sqm) → higher price\n"
                    "- Town effects vary"
                )

        st.markdown("### Comparable Price Distribution")
        norm_ft = _norm_flat_label(flat_type)
        ft_col = "flat_type_" + norm_ft.replace(" ", "_")

        comp_df = df.copy()
        if "town" in comp_df.columns:
            comp_df = comp_df[comp_df["town"] == town]
        if ft_col in comp_df.columns:
            comp_df = comp_df[comp_df[ft_col]]  # already boolean
        if "floor_area_sqm" in comp_df.columns:
            comp_df = comp_df[
                comp_df["floor_area_sqm"].between(area * 0.85, area * 1.15)
            ]
        if "remaining_lease" in comp_df.columns:
            comp_df = comp_df[
                comp_df["remaining_lease"].between(
                    max(0, lease - 10), min(99, lease + 10)
                )
            ]
        if "nearest_mrt_distance_km" in comp_df.columns:
            comp_df = comp_df[comp_df["nearest_mrt_distance_km"] <= mrt_km]

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
            )
            st.altair_chart(hist, use_container_width=True)
            st.caption(f"{len(comp_df)} comparable transactions")

        st.markdown("### Forecasted Price Trend")
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
                st.altair_chart(band + line_hist + line_pred, use_container_width=True)
            else:
                st.altair_chart(line_hist, use_container_width=True)

        st.markdown("### Recent Comparable Sales")
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

    # tips column
    with col_right:
        st.subheader("Tips")
        st.write(
            "- Start with a specific town + flat type, then adjust size/lease bands.\n"
            "- If no comps appear, widen the area or lease tolerance.\n"
            "- When your `model.pkl` is ready, the page will automatically use it for predictions."
        )
