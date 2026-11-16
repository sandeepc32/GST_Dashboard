import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import numpy as np
from style import load_css
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="GST Revenue Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)
load_css()
# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 GST Revenue Forecasting Dashboard")

st.markdown(
    """
    **Machine Learning Based GST Revenue Analytics & Forecasting**

    Analyze historical GST collections, evaluate tax relationships,
    and forecast future revenue using Linear Regression.
    """
)

st.divider()
# --------------------------------------------------
# LOAD EXCEL
# --------------------------------------------------

from modules.data_loader import (
    get_sheet_names,
    load_data
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("# GST Analytics")

st.sidebar.caption(
    "Revenue Forecasting & Tax Analytics"
)

st.sidebar.markdown("---")

sheet = st.sidebar.selectbox(
    "📂 Select Dataset",
    get_sheet_names()
)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Dashboard Modules")

st.sidebar.markdown(
    """
    🏠 **Overview**  
    📋 **Dataset**  
    📈 **Analysis**  
    🤖 **Prediction**
    """
)

st.sidebar.markdown("---")

st.sidebar.success("🟢 Dashboard Online")

st.sidebar.caption(
    "Machine Learning • Tax Analytics • Forecasting"
)
# ==========================================
# LOAD DATA
# ==========================================

with st.spinner("Loading GST dataset..."):
    df = load_data(sheet)

st.sidebar.success("✅ Dataset Loaded")

# Convert Month column into proper date
df["Month"] = pd.to_datetime(df["Month"])
# ==========================================
# DATASET SUMMARY
# ==========================================

historical_records = df["Actual_tax"].notna().sum()
forecast_records = df["Actual_tax"].isna().sum()
missing_values = df.isna().sum().sum()

start_month = df["Month"].min().strftime("%b %Y")
end_month = df["Month"].max().strftime("%b %Y")

st.markdown("## 📂 Dataset Summary")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Dataset",
    sheet
)

c2.metric(
    "Historical Months",
    historical_records
)

c3.metric(
    "Forecast Months",
    forecast_records
)

c4.metric(
    "Missing Values",
    missing_values
)

c5.metric(
    "Date Range",
    f"{start_month} → {end_month}"
)

st.divider()

# ==========================================
# DATASET HEALTH
# ==========================================

st.markdown("🩺 Dataset Health")

duplicate_rows = df.duplicated().sum()

numeric_columns = df.select_dtypes(include="number").shape[1]

total_cells = df.shape[0] * df.shape[1]

filled_cells = total_cells - missing_values

completeness = (filled_cells / total_cells) * 100

latest_actual = (
    df[df["Actual_tax"].notna()]["Month"]
    .max()
    .strftime("%b %Y")
)

h1, h2, h3, h4 = st.columns(4)

h1.metric(
    "Data Completeness",
    f"{completeness:.1f}%"
)

h2.metric(
    "Duplicate Rows",
    duplicate_rows,
    delta="Clean" if duplicate_rows == 0 else None
)

h3.metric(
    "Numeric Columns",
    numeric_columns
)

h4.metric(
    "Latest Actual GST",
    latest_actual
)

if completeness >= 99:
    st.success("✅ Dataset quality is excellent. No significant data quality issues detected.")
elif completeness >= 95:
    st.info("ℹ️ Dataset quality is good. Minor missing values detected.")
else:
    st.warning("⚠️ Dataset contains considerable missing values.")
# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Overview", "📋 Dataset", "📈 Analysis", "🤖 Prediction"]
)

# ==================================================
# OVERVIEW
# ==================================================

with tab1:

    st.markdown("📊 Dashboard Overview")

    st.info(
    f"""
    **Current Dataset:** {sheet}

    Monitor historical GST collection, identify revenue patterns,
    and evaluate the latest forecasting insights.
    """
)

    actual_tax = df["Actual_tax"].dropna()

    latest_actual = actual_tax.iloc[-1]

    highest_month = df.loc[
        df["Actual_tax"].idxmax(),
        "Month"
    ]

    
    st.divider()

    fig = px.line(
        df,
        x="Month",
        y="Actual_tax",
        markers=True,
        title="Monthly GST Collection"
    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white"
        
    )



    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.success(f"""
### 📌 Quick Insights

**Highest Collection**

📅 {highest_month.strftime('%B %Y')}

💰 {actual_tax.max():.2f} Cr
""")

    with right:
        st.info(f"""
### 📂 Dataset

Current Dataset

**{sheet}**

Total Records

**{len(df)}**
""")

    st.divider()

    # --------------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------------

    st.markdown("📖 Project Summary")

    st.markdown(
        """
        This dashboard analyzes historical GST collections and uses
        **Linear Regression** to forecast GST revenue for periods where
        actual collection data is unavailable.

        **Dashboard workflow:** Historical Data → Trend Analysis → Model Evaluation → GST Forecast
        """
    )

    st.divider()

    # --------------------------------------------------
    # QUICK PERFORMANCE SUMMARY
    # --------------------------------------------------

    st.markdown("📌 Collection Snapshot")

    valid_actual = df[["Month", "Actual_tax"]].dropna().sort_values("Month")

    if not valid_actual.empty:

        latest_row = valid_actual.iloc[-1]
        highest_row = valid_actual.loc[valid_actual["Actual_tax"].idxmax()]
        lowest_row = valid_actual.loc[valid_actual["Actual_tax"].idxmin()]

        latest_actual = latest_row["Actual_tax"]
        previous_actual = (
            valid_actual.iloc[-2]["Actual_tax"]
            if len(valid_actual) > 1
            else np.nan
        )

        if pd.notna(previous_actual) and previous_actual != 0:
            change_pct = ((latest_actual - previous_actual) / previous_actual) * 100
            change_text = f"{change_pct:+.2f}% vs previous available month"
        else:
            change_text = "No previous month available"

        k1, k2, k3, k4 = st.columns(4)

        # ==========================================
        # EXECUTIVE KPI DASHBOARD
        # ==========================================

        total_gst = valid_actual["Actual_tax"].sum()

        average_gst = valid_actual["Actual_tax"].mean()

        highest_gst = valid_actual["Actual_tax"].max()

        lowest_gst = valid_actual["Actual_tax"].min()

        latest_month = latest_row["Month"].strftime("%b %Y")

        growth = 0

        if pd.notna(previous_actual) and previous_actual != 0:
         growth = (
        (latest_actual - previous_actual)
        / previous_actual
    ) * 100

        k1, k2, k3 = st.columns(3)

        k1.metric(
    "💰 Total GST",
    f"{total_gst:.2f} Cr"
)

        k2.metric(
        "📈 Latest GST",
        f"{latest_actual:.2f} Cr",
        delta=f"{growth:+.2f}%"
)

        k3.metric(
        "📊 Average GST",
        f"{average_gst:.2f} Cr"
)

        k4, k5, k6 = st.columns(3)

        k4.metric(
        "🏆 Highest GST",
        f"{highest_gst:.2f} Cr"
)

        k5.metric(
        "📉 Lowest GST",
        f"{lowest_gst:.2f} Cr"
)

        k6.metric(
    "📅 Latest Month",
    latest_month
)

        st.divider()
        st.markdown("📈 Executive Summary")
        
        # --------------------------------------------------
        # MONTHLY GST TREND
        # --------------------------------------------------
        
        st.markdown("📈 Monthly GST Collection Trend")

        fig = px.line(
            valid_actual,
            x="Month",
            y="Actual_tax",
            markers=True,
            title="Actual GST Collection by Month"
        )

        fig.update_traces(
            line_width=3,
            marker_size=8
        )

        fig.update_layout(
            hovermode="x unified",
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="GST Revenue (Cr)",
            height=450,
        )



        st.plotly_chart(
            fig,
            use_container_width=True,
            key="overview_gst_trend"
        )

        st.divider()

        # --------------------------------------------------
        # AUTOMATED INSIGHTS
        # --------------------------------------------------

        insight_left, insight_right = st.columns(2)

        with insight_left:
            st.success(
                f"""
                ### 🧠 Quick Insights

                **Highest Collection:**  
                {highest_row["Month"].strftime("%B %Y")} — **{highest_row["Actual_tax"]:.2f} Cr**

                **Lowest Collection:**  
                {lowest_row["Month"].strftime("%B %Y")} — **{lowest_row["Actual_tax"]:.2f} Cr**

                **Average Collection:**  
                **{valid_actual["Actual_tax"].mean():.2f} Cr**
                """
            )

        with insight_right:
            st.info(
                f"""
                ### 📂 Dataset Status

                **Current Dataset:**  
                **{sheet}**

                **Historical Records:**  
                **{len(valid_actual):,}**

                **Records Awaiting Forecast:**  
                **{df["Actual_tax"].isna().sum():,}**

                **Latest Available Month:**  
                **{latest_row["Month"].strftime("%B %Y")}**
                """
            )

        st.divider()

        # --------------------------------------------------
        # PROJECT FEATURES
        # --------------------------------------------------

        st.markdown("🚀 Dashboard Capabilities")

        feature1, feature2, feature3, feature4 = st.columns(4)

        feature1.info(
            "📊 **Historical Analysis**\n\n"
            "Monitor monthly GST collection trends."
        )

        feature2.info(
            "🔎 **Tax Analytics**\n\n"
            "Compare Outward, Inward and Difference Tax."
        )

        feature3.info(
            "🤖 **ML Forecasting**\n\n"
            "Predict GST revenue using Linear Regression."
        )

        feature4.info(
            "📈 **Model Evaluation**\n\n"
            "Compare R², MAE, RMSE and MAPE."
        )

    else:
        st.warning("⚠️ No valid Actual_tax records are available for the overview.")

# ==================================================
# DATASET
# ==================================================

with tab2:

    st.markdown("GST Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # DOWNLOAD DATASET
    # --------------------------------------------------

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Dataset CSV",
        data=csv_data,
        file_name=f"{sheet}_GST_Dataset.csv",
        mime="text/csv",
        key="download_dataset_csv"
    )

# ==================================================
# ANALYSIS
# ==================================================

with tab3:

    st.markdown("📈 GST Data Analysis")
    
    st.markdown(
        """
        Explore historical GST collection patterns, tax components,
        and relationships between key revenue indicators.
        """
    )

    # --------------------------------------------------
    # PREPARE ANALYSIS DATA
    # --------------------------------------------------

    analysis_df = df.copy()

    numeric_columns = [
        "Outward_tax",
        "Inward_tax",
        "Difference_tax",
        "Actual_tax"
    ]

    for col in numeric_columns:
        analysis_df[col] = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # ANALYTICS SUMMARY
    # --------------------------------------------------

    avg_actual = analysis_df["Actual_tax"].mean()
    avg_outward = analysis_df["Outward_tax"].mean()
    avg_inward = analysis_df["Inward_tax"].mean()
    available_records = analysis_df["Actual_tax"].notna().sum()

        # ==========================================
        # EXECUTIVE ANALYTICS SUMMARY
        # ==========================================

    avg_actual = analysis_df["Actual_tax"].mean()
    median_actual = analysis_df["Actual_tax"].median()
    max_actual = analysis_df["Actual_tax"].max()
    std_actual = analysis_df["Actual_tax"].std()

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
    "💰 Mean GST",
    f"{avg_actual:.2f} Cr"
)

    a2.metric(
    "📍 Median GST",
    f"{median_actual:.2f} Cr"
)

    a3.metric(
    "🏆 Highest GST",
    f"{max_actual:.2f} Cr"
)

    a4.metric(
    "📊 Standard Deviation",
    f"{std_actual:.2f}"
)

    st.divider()
        # --------------------------------------------------
    # GST DISTRIBUTION
    # --------------------------------------------------

    st.markdown("📊 GST Distribution")

    distribution_df = analysis_df[
    ["Actual_tax"]
].dropna()

    distribution_fig = px.histogram(
    distribution_df,
    x="Actual_tax",
    nbins=10,
    title="Distribution of Historical GST Collection"
)

    distribution_fig.update_layout(
    template="plotly_white",
    xaxis_title="GST Collection (Cr)",
    yaxis_title="Number of Months",
    bargap=0.08,
    height=450
)
    
    
    

    st.plotly_chart(
    distribution_fig,
    use_container_width=True,
    key="gst_distribution"
)

    st.caption(
    "This histogram shows how GST collections are distributed across the historical dataset."
)

    st.divider()
    # --------------------------------------------------
    # TAX COMPONENT TREND
    # --------------------------------------------------

    st.markdown("📊 GST Tax Components Trend")

    component_df = analysis_df[
        [
            "Month",
            "Outward_tax",
            "Inward_tax",
            "Difference_tax"
        ]
    ].melt(
        id_vars="Month",
        var_name="Tax Component",
        value_name="Tax Value"
    )

    component_df["Tax Component"] = component_df[
        "Tax Component"
    ].replace(
        {
            "Outward_tax": "Outward Tax",
            "Inward_tax": "Inward Tax",
            "Difference_tax": "Difference Tax"
        }
    )

    component_fig = px.line(
        component_df,
        x="Month",
        y="Tax Value",
        color="Tax Component",
        markers=True,
        title="Monthly Tax Component Movement"
    )

    component_fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Tax Value (Cr)",
        height=450
    )
        
        
        

    component_fig.update_traces(
        line_width=3,
        marker_size=7
    )

    st.plotly_chart(
        component_fig,
        use_container_width=True,
        key="analysis_tax_components"
    )

    st.divider()
        # --------------------------------------------------
        # GST OUTLIER ANALYSIS
        # --------------------------------------------------

    st.markdown("📦 GST Outlier Detection")

    box_fig = px.box(
    analysis_df,
    y="Actual_tax",
    points="all",
    title="Actual GST Collection Distribution"
)

    box_fig.update_layout(
    template="plotly_white",
    yaxis_title="GST Collection (Cr)",
    height=450
)
        
        
        

    st.plotly_chart(
    box_fig,
    use_container_width=True,
    key="gst_boxplot"
)

    st.caption(
    "The box plot highlights the spread of GST collections and helps identify any unusually high or low collection months."
)

    st.divider()
    # --------------------------------------------------
    # ACTUAL GST TREND
    # --------------------------------------------------

    st.markdown("📈 Actual GST Collection Trend")

    actual_analysis = analysis_df[
        ["Month", "Actual_tax"]
    ].dropna()

    actual_fig = px.line(
        actual_analysis,
        x="Month",
        y="Actual_tax",
        markers=True,
        title="Historical GST Collection"
    )

    actual_fig.update_traces(
        line_width=3,
        marker_size=8
    )

    actual_fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Actual GST (Cr)",
        height=450
    )
        
        
        

    st.plotly_chart(
        actual_fig,
        use_container_width=True,
        key="analysis_actual_gst"
    )

    st.divider()

    # --------------------------------------------------
    # CORRELATION ANALYSIS
    # --------------------------------------------------

    st.markdown("🔗 Tax Component Correlation")

    correlation_df = analysis_df[
        [
            "Outward_tax",
            "Inward_tax",
            "Difference_tax",
            "Actual_tax"
        ]
    ].corr()

    correlation_fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    sns.heatmap(
        correlation_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Correlation Between GST Variables",
        fontsize=14,
        pad=15
    )

    st.pyplot(
        correlation_fig,
        use_container_width=True
    )
        # ==========================================
# CORRELATION INSIGHTS
# ==========================================

corr_actual = correlation_df["Actual_tax"].drop("Actual_tax")

highest_feature = corr_actual.idxmax()
highest_value = corr_actual.max()

lowest_feature = corr_actual.idxmin()
lowest_value = corr_actual.min()

st.markdown("### 🧠 Correlation Insights")

left, right = st.columns(2)

with left:

    st.success(
        f"""
**Strongest Predictor**

📌 Feature: **{highest_feature.replace("_", " ")}**

📈 Correlation: **{highest_value:.3f}**
"""
    )

with right:

    st.info(
        f"""
**Weakest Predictor**

📌 Feature: **{lowest_feature.replace("_", " ")}**

📉 Correlation: **{lowest_value:.3f}**
"""
    )
    st.info(
        """
        **💡 Interpretation:** Correlation values closer to **+1**
        indicate a strong positive relationship, while values closer
        to **-1** indicate a strong negative relationship.
        """
    )


# ==================================================
# PREDICTION
# ==================================================

with tab4:

    st.markdown("🤖 GST Revenue Prediction")

    st.markdown(
        """
        ### Machine Learning Forecasting Engine

        Month N e-invoice indicators (Outward Tax, Inward Tax and
        Difference Tax) are used to forecast Month N+1 official
        Gross Domestic Revenue.

        Once the official actual becomes available, the historical
        forecast is evaluated and the new actual is automatically
        included in subsequent model retraining.
        """
    )

    st.caption(
        "Leakage-free one-month-ahead forecasting with chronological "
        "walk-forward validation."
    )

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    required_columns = [
        "Month",
        "Outward_tax",
        "Inward_tax",
        "Difference_tax",
        "Actual_tax",
    ]

    missing_columns = [c for c in required_columns if c not in df.columns]

    if missing_columns:
        st.error(
            "❌ Missing required columns: "
            + ", ".join(missing_columns)
        )
    else:

        work_df = df[required_columns].copy()
        work_df["Month"] = pd.to_datetime(
            work_df["Month"], errors="coerce"
        )

        for col in required_columns[1:]:
            work_df[col] = pd.to_numeric(
                work_df[col], errors="coerce"
            )

        work_df = (
            work_df
            .dropna(subset=["Month"])
            .sort_values("Month")
            .drop_duplicates(subset=["Month"], keep="last")
            .reset_index(drop=True)
        )

        features = [
            "Outward_tax",
            "Inward_tax",
            "Difference_tax",
        ]
        target = "Actual_tax"

        # ==================================================
        # CRITICAL ALIGNMENT
        #
        # Month N e-invoice data -> Month N+1 GST actual
        # ==================================================

        aligned_df = work_df.copy()
        aligned_df["Target_Month"] = aligned_df["Month"].shift(-1)
        aligned_df["Target_GST"] = aligned_df[target].shift(-1)

        # Historical supervised pairs: the following month's actual
        # exists and all Month N e-invoice indicators are available.
        aligned_history = aligned_df[
            aligned_df[features].notna().all(axis=1)
            & aligned_df["Target_GST"].notna()
        ].copy()

        # Current/future candidates: Month N inputs exist, but the
        # following month's official GST actual does not.
        forecast_candidates = aligned_df[
            aligned_df[features].notna().all(axis=1)
            & aligned_df["Target_GST"].isna()
        ].copy()

        # ==================================================
        # METRIC HELPERS
        # ==================================================

        def calculate_mape(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            mask = (
                np.isfinite(y_true)
                & np.isfinite(y_pred)
                & (y_true != 0)
            )

            if not mask.any():
                return np.nan

            return float(
                np.mean(
                    np.abs(
                        (y_true[mask] - y_pred[mask])
                        / y_true[mask]
                    )
                ) * 100
            )

        def calculate_metrics(y_true, y_pred):
            return {
                "R²": r2_score(y_true, y_pred),
                "MAE (Cr)": mean_absolute_error(y_true, y_pred),
                "RMSE (Cr)": np.sqrt(
                    mean_squared_error(y_true, y_pred)
                ),
                "MAPE (%)": calculate_mape(y_true, y_pred),
            }

        # ==================================================
        # WALK-FORWARD VALIDATION
        # ==================================================

        def walk_forward_evaluate(
            data,
            selected_features,
            min_train_size=4
        ):
            data = (
                data
                .sort_values("Target_Month")
                .reset_index(drop=True)
            )

            results = []

            if len(data) <= min_train_size:
                return pd.DataFrame()

            for i in range(min_train_size, len(data)):

                training = data.iloc[:i].copy()
                test_row = data.iloc[[i]].copy()

                training = training.dropna(
                    subset=selected_features + ["Target_GST"]
                )
                test_row = test_row.dropna(
                    subset=selected_features + ["Target_GST"]
                )

                if training.empty or test_row.empty:
                    continue

                model = LinearRegression()

                model.fit(
                    training[selected_features],
                    training["Target_GST"]
                )

                prediction = float(
                    model.predict(
                        test_row[selected_features]
                    )[0]
                )

                actual = float(test_row["Target_GST"].iloc[0])

                error = (
                    abs(prediction - actual)
                    / abs(actual)
                    * 100
                    if actual != 0
                    else np.nan
                )

                results.append(
                    {
                        "Source Month": test_row["Month"].iloc[0],
                        "Forecast Month": test_row["Target_Month"].iloc[0],
                        "Predicted GST": prediction,
                        "Actual GST": actual,
                        "Error (%)": error,
                    }
                )

            return pd.DataFrame(results)

        # ==================================================
        # DATA STATUS
        # ==================================================

        st.markdown("### 📅 Forecasting Data Alignment")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Historical Forecast Pairs",
            f"{len(aligned_history):,}"
        )

        s2.metric(
            "Current Forecast Candidates",
            f"{len(forecast_candidates):,}"
        )

        s3.metric(
            "E-Invoice Months",
            f"{work_df[features].notna().all(axis=1).sum():,}"
        )

        s4.metric(
            "Actual GST Months",
            f"{work_df[target].notna().sum():,}"
        )

        st.info(
            "📌 **Business alignment:** Month N e-invoice data "
            "predicts Month N+1 official Gross Domestic Revenue."
        )

        if len(aligned_history) < 5:

            st.error(
                "❌ Not enough aligned historical observations "
                "for reliable walk-forward validation."
            )

        else:

            # ==================================================
            # MODEL COMPARISON
            # ==================================================

            st.markdown("### 🧪 Walk-Forward Model Comparison")

            st.caption(
                "Every historical forecast is generated using only "
                "data available before that forecast month."
            )

            model_features = {
                "Linear Regression - All Tax Indicators": features,
                "Linear Regression - Outward Only": ["Outward_tax"],
                "Linear Regression - Outward + Inward": [
                    "Outward_tax",
                    "Inward_tax",
                ],
            }

            comparison_results = []
            walk_forward_results = {}

            min_train_size = max(
                4,
                min(8, len(aligned_history) - 1)
            )

            for model_name, selected_features in model_features.items():

                wf = walk_forward_evaluate(
                    aligned_history,
                    selected_features,
                    min_train_size=min_train_size,
                )

                walk_forward_results[model_name] = wf

                if wf.empty:
                    continue

                metrics = calculate_metrics(
                    wf["Actual GST"],
                    wf["Predicted GST"],
                )

                comparison_results.append(
                    {
                        "Model": model_name,
                        **metrics,
                        "Forecasts Evaluated": len(wf),
                    }
                )

            comparison_df = pd.DataFrame(comparison_results)

            if comparison_df.empty:

                st.error(
                    "❌ Walk-forward validation could not be completed."
                )

            else:

                comparison_df = (
                    comparison_df
                    .sort_values("MAPE (%)", na_position="last")
                    .reset_index(drop=True)
                )

                best_model_name = comparison_df.iloc[0]["Model"]
                best_features = model_features[best_model_name]
                best_wf = walk_forward_results[best_model_name]

                st.markdown("### 🏆 Best Walk-Forward Model")

                b1, b2, b3, b4 = st.columns(4)

                b1.success(
                    f"""
                    **Selected Model**

                    ### {best_model_name}

                    Features: **{", ".join(best_features).replace("_", " ")}**
                    """
                )

                b2.metric(
                    "Walk-Forward MAPE",
                    f"{comparison_df.iloc[0]['MAPE (%)']:.2f}%"
                )

                b3.metric(
                    "MAE",
                    f"{comparison_df.iloc[0]['MAE (Cr)']:.3f} Cr"
                )

                b4.metric(
                    "Forecasts Evaluated",
                    f"{int(comparison_df.iloc[0]['Forecasts Evaluated']):,}"
                )

                st.markdown("### 📋 Model Performance")

                display_df = comparison_df.copy()
                display_df["R²"] = display_df["R²"].round(3)
                display_df["MAE (Cr)"] = display_df["MAE (Cr)"].round(3)
                display_df["RMSE (Cr)"] = display_df["RMSE (Cr)"].round(3)
                display_df["MAPE (%)"] = display_df["MAPE (%)"].round(2)

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # ==================================================
                # WALK-FORWARD ACTUAL VS PREDICTED
                # ==================================================

                st.markdown("### 📈 Walk-Forward Actual vs Predicted")

                if not best_wf.empty:

                    evaluation_long = best_wf[
                        [
                            "Forecast Month",
                            "Actual GST",
                            "Predicted GST",
                        ]
                    ].melt(
                        id_vars="Forecast Month",
                        value_vars=[
                            "Actual GST",
                            "Predicted GST",
                        ],
                        var_name="Type",
                        value_name="GST",
                    )

                    wf_fig = px.line(
                        evaluation_long,
                        x="Forecast Month",
                        y="GST",
                        color="Type",
                        markers=True,
                        title=(
                            f"{best_model_name} — "
                            "Walk-Forward Validation"
                        ),
                    )

                    wf_fig.update_layout(
                        hovermode="x unified",
                        template="plotly_white",
                        xaxis_title="Forecast Month",
                        yaxis_title="GST Revenue (Cr)",
                        height=450,
                    )

                    st.plotly_chart(
                        wf_fig,
                        use_container_width=True,
                        key="walk_forward_actual_vs_predicted",
                    )

                    st.markdown("### 📊 Month-by-Month Forecast Performance")

                    performance_display = best_wf.copy()

                    performance_display["Source Month"] = (
                        performance_display["Source Month"]
                        .dt.strftime("%B %Y")
                    )

                    performance_display["Forecast Month"] = (
                        performance_display["Forecast Month"]
                        .dt.strftime("%B %Y")
                    )

                    for col in [
                        "Predicted GST",
                        "Actual GST",
                        "Error (%)",
                    ]:
                        performance_display[col] = (
                            performance_display[col].round(3)
                        )

                    st.dataframe(
                        performance_display,
                        use_container_width=True,
                        hide_index=True,
                    )

                # ==================================================
                # TRAIN FINAL MODEL ON ALL AVAILABLE ALIGNED DATA
                # ==================================================

                final_model = LinearRegression()

                final_model.fit(
                    aligned_history[best_features],
                    aligned_history["Target_GST"],
                )

                coefficients = final_model.coef_
                intercept = float(final_model.intercept_)

                # ==================================================
                # CURRENT FORECAST
                # ==================================================

                st.divider()
                st.markdown("### 🔮 Current / Upcoming GST Forecast")

                if forecast_candidates.empty:

                    st.success(
                        "✅ No current forecast candidate is available. "
                        "Add the latest month's e-invoice values with a "
                        "blank Actual_tax to generate the next forecast."
                    )

                else:

                    latest_forecast_row = (
                        forecast_candidates
                        .sort_values("Month")
                        .iloc[-1]
                    )

                    source_month = latest_forecast_row["Month"]
                    forecast_month = latest_forecast_row["Target_Month"]

                    # If this is the final row in the dataset, there is
                    # no following row yet. The forecast target is therefore
                    # the next calendar month.
                    if pd.isna(forecast_month):
                        forecast_month = (
                            source_month
                            + pd.DateOffset(months=1)
                        )

                    current_prediction = float(
                        final_model.predict(
                            latest_forecast_row[
                                best_features
                            ].to_frame().T
                        )[0]
                    )

                    st.markdown(
                        f"### 🎯 {forecast_month.strftime('%B %Y')}"
                    )

                    st.caption(
                        f"Generated from {source_month.strftime('%B %Y')} "
                        "e-invoice data."
                    )

                    f1, f2, f3 = st.columns(3)

                    f1.success(
                        f"""
                        ## ₹{current_prediction:.2f} Cr

                        **Predicted GST Collection**
                        """
                    )

                    f2.metric(
                        "Source E-Invoice Month",
                        source_month.strftime("%B %Y"),
                    )

                    f3.metric(
                        "Status",
                        "Awaiting Actual",
                    )

                    input_cols = st.columns(len(features))

                    for idx, feature in enumerate(features):

                        input_cols[idx].metric(
                            feature.replace("_", " "),
                            f"{latest_forecast_row[feature]:.3f} Cr",
                        )

                    st.info(
                        f"📌 When the official {forecast_month.strftime('%B %Y')} "
                        "Gross Domestic Revenue is entered as Actual_tax, "
                        "that month becomes part of the historical training "
                        "set for the next forecast."
                    )

                # ==================================================
                # MODEL EQUATION
                # ==================================================

                st.divider()
                st.markdown("### 🧮 Final Regression Equation")

                terms = []

                for feature, coefficient in zip(
                    best_features,
                    coefficients,
                ):
                    terms.append(
                        f"{coefficient:+.4f} × "
                        f"{feature.replace('_', ' ')}"
                    )

                equation_text = (
                    f"GST = {intercept:.4f} "
                    + " ".join(terms)
                )

                st.success(f"**{equation_text}**")

                coefficient_df = pd.DataFrame(
                    {
                        "Feature": [
                            feature.replace("_", " ")
                            for feature in best_features
                        ],
                        "Coefficient": coefficients,
                    }
                )

                coefficient_fig = px.bar(
                    coefficient_df,
                    x="Feature",
                    y="Coefficient",
                    text="Coefficient",
                    title="Final Model Feature Coefficients",
                )

                coefficient_fig.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                )

                coefficient_fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Feature",
                    yaxis_title="Coefficient",
                    showlegend=False,
                    height=400,
                )

                st.plotly_chart(
                    coefficient_fig,
                    use_container_width=True,
                    key="forecast_model_coefficients",
                )

                # ==================================================
                # DOWNLOAD PERFORMANCE
                # ==================================================

                if not best_wf.empty:

                    history_download = best_wf.copy()

                    history_download["Source Month"] = (
                        history_download["Source Month"]
                        .dt.strftime("%B %Y")
                    )

                    history_download["Forecast Month"] = (
                        history_download["Forecast Month"]
                        .dt.strftime("%B %Y")
                    )

                    history_csv = history_download.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="📥 Download Forecast Performance CSV",
                        data=history_csv,
                        file_name="GST_Forecast_Performance.csv",
                        mime="text/csv",
                        key="download_forecast_performance",
                    )

                # ==================================================
                # CURRENT FORECAST DOWNLOAD
                # ==================================================

                if not forecast_candidates.empty:

                    forecast_export = pd.DataFrame(
                        [
                            {
                                "E-Invoice Month": source_month.strftime(
                                    "%B %Y"
                                ),
                                "Forecast Month": forecast_month.strftime(
                                    "%B %Y"
                                ),
                                "Outward Tax (Cr)": latest_forecast_row[
                                    "Outward_tax"
                                ],
                                "Inward Tax (Cr)": latest_forecast_row[
                                    "Inward_tax"
                                ],
                                "Difference Tax (Cr)": latest_forecast_row[
                                    "Difference_tax"
                                ],
                                "Predicted GST (Cr)": current_prediction,
                                "Model": best_model_name,
                            }
                        ]
                    )

                    forecast_csv = forecast_export.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="📥 Download Current GST Forecast CSV",
                        data=forecast_csv,
                        file_name="Current_GST_Forecast.csv",
                        mime="text/csv",
                        key="download_current_gst_forecast",
                    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "📊 GST Revenue Forecasting Dashboard • "
    "Machine Learning & Tax Analytics"
)

st.caption("Developed by Sandeep C •  National Informatics Centre • Year 2026")
