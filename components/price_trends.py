import streamlit as st
import plotly.express as px

def show_price_trends(df):
    """Display price trends over time with user controls."""

    # Labels
    measure_labels = {
        "resale_price": "Resale Price",
        "price_per_sqm": "Price per sqm"
    }
    group_by_labels = {
        "region": "Region",
        "flat_type": "Flat Type",
        "flat_model": "Flat Model"
    }

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        y_axis = st.selectbox(
            "Measure",
            list(measure_labels.keys()),
            format_func=lambda x: measure_labels[x],
            index=0,
            key="price_trend_measure"
        )

    with col2:
        group_by = st.selectbox(
            "Group by",
            list(group_by_labels.keys()),
            format_func=lambda x: group_by_labels[x],
            index=0,
            key="price_trend_group"
        )

    with col3:
        agg_func = st.selectbox(
            "Aggregation",
            ["Mean", "Median", "Min", "Max"],
            index=0,
            key="price_trend_agg"
        )

    # --- Transform resale_price into thousands ---
    plot_df = df.copy()
    if y_axis == "resale_price":
        plot_df[y_axis] = plot_df[y_axis] / 1000  # convert to '000s

    # --- Aggregate data ---
    if agg_func == "Mean":
        agg_df = plot_df.groupby(["month", group_by])[y_axis].mean().reset_index()
    elif agg_func == "Median":
        agg_df = plot_df.groupby(["month", group_by])[y_axis].median().reset_index()
    elif agg_func == "Min":
        agg_df = plot_df.groupby(["month", group_by])[y_axis].min().reset_index()
    elif agg_func == "Max":
        agg_df = plot_df.groupby(["month", group_by])[y_axis].max().reset_index()

    agg_df = agg_df.sort_values("month")

    # --- Plot ---
    fig = px.line(
        agg_df,
        x="month",
        y=y_axis,
        color=group_by,
        template="plotly_white",
        title=f"{agg_func} {measure_labels[y_axis]} by {group_by_labels[group_by]} (2017–2025)",
        labels={y_axis: measure_labels[y_axis], "month": "Month"},
    )

    # Axis formatting
    if y_axis == "resale_price":
        fig.update_yaxes(tickprefix="$", ticksuffix="k")

    fig.update_layout(
        title=dict(
            text=f"{agg_func} {measure_labels[y_axis]} by {group_by_labels[group_by]} (2017–2025)",
            x=0,
            xanchor="left",
            font=dict(size=20, color="black")
        ),
        xaxis_title="Year",
        yaxis_title=f"{agg_func} {measure_labels[y_axis]}",
        legend_title=group_by_labels[group_by],
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
