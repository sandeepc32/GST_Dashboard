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
    font=dict(
    size=14,
    color="#374151"
),

    title_x=0,

    paper_bgcolor="white",

    plot_bgcolor="white"
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
        font=dict(
    size=14,
    color="#374151"
),

        title_x=0,

        paper_bgcolor="white",

        plot_bgcolor="white"
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
    font=dict(
        size=14,
        color="#374151"
    ),
    
    title_x=0,
    
    paper_bgcolor="white",
    
    plot_bgcolor="white"

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
    font=dict(
            size=14,
            color="#374151"
        ),
        
    title_x=0,
        
    paper_bgcolor="white",
        
    plot_bgcolor="white"

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
    font=dict(
            size=14,
            color="#374151"
        ),
        
    title_x=0,
        
    paper_bgcolor="white",
        
    plot_bgcolor="white"

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
    font=dict(
            size=14,
            color="#374151"
        ),
        
    title_x=0,
        
    paper_bgcolor="white",
        
    plot_bgcolor="white"

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

    This module evaluates multiple Linear Regression models,
    selects the best-performing feature combination, and estimates
    GST revenue for the latest month when official collection data
    is not yet available.

    Once the official actual value becomes available, the dashboard
    compares the previous forecast with the actual collection and
    automatically moves that month into historical performance.
    """
)

    st.caption(
    "Model selection is based on chronological 80/20 train-test evaluation "
    "using MAPE as the primary selection metric."
)
# ==================================================
# ABOUT THE MODEL
# ==================================================

    st.markdown("🧠 About the Model")

    info1, info2, info3 = st.columns(3)

    with info1:

        st.info(
        """
    **Algorithm**

Linear Regression

**Library**

Scikit-learn
"""
    )

    with info2:

        st.info(
        """
**Validation**

Chronological Split

80% Train

20% Test
"""
    )

    with info3:

        st.info(
        """
**Selection Metric**

MAPE

(Lower is Better)
"""
    )

    st.caption(
    "The forecasting engine automatically evaluates multiple feature combinations and selects the model with the lowest testing MAPE."
)

    st.divider()


    # --------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------

    features = [
        "Outward_tax",
        "Inward_tax",
        "Difference_tax"
    ]

    target = "Actual_tax"

    for col in features + [target]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # HISTORICAL / FUTURE DATA
    # --------------------------------------------------

    train_df = df.dropna(
        subset=features + [target]
    ).copy()

    train_df = train_df.sort_values(
        "Month"
    ).reset_index(drop=True)

    predict_df = df[
        df[target].isna()
    ].dropna(
        subset=features
    ).copy()

    if len(train_df) < 10:

        st.error(
            "❌ Not enough historical data for model evaluation."
        )

    else:

        # ==================================================
        # MODEL COMPARISON
        # ==================================================

        st.markdown("🧪 Model Comparison")

        st.markdown(
            """
            Four feature combinations are evaluated using the same
            **80/20 chronological train-test split**.
            """
        )

        split_index = int(
            len(train_df) * 0.80
        )

        training_data = train_df.iloc[
            :split_index
        ].copy()

        testing_data = train_df.iloc[
            split_index:
        ].copy()

        y_train = training_data[target]
        y_test = testing_data[target]

        model_features = {
            "Model 1 - Outward Tax": [
                "Outward_tax"
            ],
            "Model 2 - Difference Tax": [
                "Difference_tax"
            ],
            "Model 3 - Outward + Inward": [
                "Outward_tax",
                "Inward_tax"
            ],
            "Model 4 - All Features": [
                "Outward_tax",
                "Inward_tax",
                "Difference_tax"
            ]
        }

        comparison_results = []
        # Store model objects and predictions
        trained_models = {}

        for model_name, selected_features in model_features.items():

            comparison_model = LinearRegression()

            comparison_model.fit(
                training_data[selected_features],
                y_train
            )
            
            model_predictions = comparison_model.predict(
                testing_data[selected_features]
            )
            trained_models[model_name] = {
            "model": comparison_model,
            "predictions": model_predictions,
            "features": selected_features
}
            model_r2 = r2_score(
                y_test,
                model_predictions
            )

            model_mae = mean_absolute_error(
                y_test,
                model_predictions
            )

            model_rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    model_predictions
                )
            )

            non_zero = y_test != 0

            if non_zero.any():

                model_mape = np.mean(
                    np.abs(
                        (
                            y_test[non_zero]
                            - model_predictions[non_zero]
                        )
                        / y_test[non_zero]
                    )
                ) * 100

            else:

                model_mape = np.nan

            comparison_results.append(
                {
                    "Model": model_name,
                    "R²": model_r2,
                    "MAE (Cr)": model_mae,
                    "RMSE (Cr)": model_rmse,
                    "MAPE (%)": model_mape
                }
            )

        comparison_df = pd.DataFrame(
            comparison_results
        )

        best_model_row = comparison_df.loc[
            comparison_df["MAPE (%)"].idxmin()
        ]

        best_model_name = best_model_row["Model"]
        best_predictions = trained_models[
        best_model_name
        ]["predictions"]

        best_model = trained_models[
        best_model_name
        ]["model"]
        residuals = y_test - best_predictions
        best_features = model_features[best_model_name]
        # ==========================================
# STORE BEST MODEL DETAILS
# ==========================================

        best_model = trained_models[best_model_name]["model"]

        best_predictions = trained_models[best_model_name]["predictions"]

        residuals = y_test - best_predictions

        # --------------------------------------------------
        # MODEL COMPARISON CARDS
        # --------------------------------------------------

        st.markdown("### 🏆 Best Model")

        best_col1, best_col2, best_col3, best_col4 = st.columns(4)

        with best_col1:
            st.success(
            f"""
            **Best Model**

        ### {best_model_name}

        Features: **{", ".join(best_features).replace("_", " ")}**
        """
    )

        with best_col2:
            st.metric(
                "R² Score",
                f"{best_model_row['R²']:.3f}"
            )

        with best_col3:
            st.metric(
                "MAPE",
                f"{best_model_row['MAPE (%)']:.2f}%"
            )
        with best_col4:
            st.metric(
            "Training Records",
            f"{len(training_data):,}"
)

        st.caption(
        "The model with the lowest test MAPE is selected as the final forecasting model."
)

        
        st.markdown("### 📋 Model Performance Comparison")

        st.caption(
    "All models use the same chronological 80/20 train-test split "
    "to ensure a fair comparison."
)

        display_df = comparison_df.copy()

        display_df["R²"] = display_df["R²"].round(3)
        display_df["MAE (Cr)"] = display_df["MAE (Cr)"].round(3)
        display_df["RMSE (Cr)"] = display_df["RMSE (Cr)"].round(3)
        display_df["MAPE (%)"] = display_df["MAPE (%)"].round(2)

        st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ------------------------------
# Model Comparison Bar Chart
# ------------------------------

        st.markdown("### 📊 Model Comparison (MAPE %)")

        comparison_chart = px.bar(
    comparison_df.sort_values("MAPE (%)"),
    x="Model",
    y="MAPE (%)",
    text="MAPE (%)",
    color="MAPE (%)",
    title="Model Comparison Based on Test MAPE"
)

        comparison_chart.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

        comparison_chart.update_layout(
    template="plotly_white",
    xaxis_title="Models",
    yaxis_title="MAPE (%)",
    coloraxis_showscale=False,
    height=500
)

        st.plotly_chart(
    comparison_chart,
    use_container_width=True,
    key="model_comparison_bar"
)

        st.info(
    "Lower MAPE indicates better forecasting accuracy. "
    "The model with the lowest MAPE is selected as the final forecasting model."
)

        st.divider()

        # ==================================================
        # SELECTED MODEL PERFORMANCE
        # ==================================================

        st.markdown("📊 Selected Model Performance")

        best_test_model = LinearRegression()

        best_test_model.fit(
            training_data[best_features],
            y_train
        )

        best_test_predictions = best_test_model.predict(
            testing_data[best_features]
        )

        best_r2 = r2_score(
            y_test,
            best_test_predictions
        )

        best_mae = mean_absolute_error(
            y_test,
            best_test_predictions
        )

        best_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                best_test_predictions
            )
        )

        non_zero = y_test != 0

        if non_zero.any():

            best_mape = np.mean(
                np.abs(
                    (
                        y_test[non_zero]
                        - best_test_predictions[non_zero]
                    )
                    / y_test[non_zero]
                )
            ) * 100

        else:

            best_mape = np.nan

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "R² Score",
            f"{best_r2:.3f}"
        )

        p2.metric(
            "MAE",
            f"{best_mae:.3f} Cr"
        )

        p3.metric(
            "RMSE",
            f"{best_rmse:.3f} Cr"
        )

        p4.metric(
            "MAPE",
            f"{best_mape:.2f}%"
        )

        
        st.info(
        f"""
    ### 💡 Why was this model selected?

    **{best_model_name}** achieved the lowest test MAPE among all evaluated
    feature combinations.

    The model was evaluated using a chronological train-test split, meaning
    earlier observations were used for training while the latest historical
    observations were reserved for testing.

    This approach provides a more realistic estimate of how the model may
    perform when forecasting future GST revenue.
    """
)



        st.divider()
        # ==========================================
# MODEL DIAGNOSTICS
# ==========================================

        st.markdown("📉 Model Diagnostics")

        diagnostic_df = pd.DataFrame({
    "Predicted GST": best_predictions,
    "Residual Error": residuals
})

        diagnostic_fig = px.scatter(
    diagnostic_df,
    x="Predicted GST",
    y="Residual Error",
    title="Residual Plot",
    trendline="ols"
)

        diagnostic_fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="red"
)

        diagnostic_fig.update_layout(
    template="plotly_white",
    height=500
)
        font=dict(
    size=14,
    color="#374151"
                ),
                
        title_x=0,
                
        paper_bgcolor="white",
                
        plot_bgcolor="white"

        st.plotly_chart(
    diagnostic_fig,
    use_container_width=True,
    key="residual_plot"
)

        st.caption(
    "Residuals randomly scattered around zero indicate that the Linear Regression model fits the data reasonably well."
)

        st.divider()
        st.markdown("### 🩺 Diagnostic Summary")

        mean_error = residuals.mean()

        max_error = residuals.abs().max()

        c1, c2 = st.columns(2)

        c1.metric(
    "Mean Residual",
    f"{mean_error:.4f}"
)

        c2.metric(
    "Maximum Error",
    f"{max_error:.4f}"
)
    if abs(mean_error) < 0.05:
        st.success(
        "✅ Residuals are centered around zero, indicating minimal prediction bias."
    )
    else:
        st.warning(
        "⚠ Residuals show noticeable bias. Review feature selection or model assumptions."
    )
        # ==================================================
# PREDICTION CONFIDENCE
# ==================================================

        st.markdown("🏅 Prediction Confidence")

    if best_mape < 5:
        st.success(f"""
### ★★★★★ Excellent

**Prediction Confidence:** Excellent

**Testing MAPE:** {best_mape:.2f}%

This model demonstrates excellent predictive accuracy and is highly suitable for forecasting GST revenue.
""")

    elif best_mape < 10:
        st.info(f"""
### ★★★★☆ Good

**Prediction Confidence:** Good

**Testing MAPE:** {best_mape:.2f}%

This model provides reliable forecasts with good predictive accuracy.
""")

    elif best_mape < 20:
        st.warning(f"""
### ★★★☆☆ Acceptable

**Prediction Confidence:** Acceptable

**Testing MAPE:** {best_mape:.2f}%

The model is suitable for forecasting, though prediction errors are comparatively higher.
""")

    else:
        st.error(f"""
### ★★☆☆☆ Low

**Prediction Confidence:** Low

**Testing MAPE:** {best_mape:.2f}%

The model requires improvement before being used for reliable forecasting.
""")

    with st.expander("📖 Confidence Scale"):
        st.markdown("""
| MAPE | Confidence |
|------:|:-----------|
| < 5% | ★★★★★ Excellent |
| 5–10% | ★★★★☆ Good |
| 10–20% | ★★★☆☆ Acceptable |
| > 20% | ★★☆☆☆ Needs Improvement |
""")

        st.divider()
        # ==================================================
        # MODEL INSIGHTS
        # ==================================================

        st.markdown("🧠 Model Insights")
        st.caption(
    "Understand the mathematical relationship learned by the selected "
    "Linear Regression model."
)

        final_model = LinearRegression()

        final_model.fit(
            train_df[best_features],
            train_df[target]
        )

        coefficients = final_model.coef_
        intercept = float(
            final_model.intercept_
        )

        if len(best_features) == 1:

            coefficient = float(
                coefficients[0]
            )

            if coefficient >= 0:

                equation_text = (
                    f"Actual GST = {intercept:.4f} + "
                    f"{coefficient:.4f} × "
                    f"{best_features[0].replace('_', ' ')}"
                )

            else:

                equation_text = (
                    f"Actual GST = {intercept:.4f} − "
                    f"{abs(coefficient):.4f} × "
                    f"{best_features[0].replace('_', ' ')}"
                )

            st.markdown("### 🧮 Regression Equation")

            st.success(
                f"**{equation_text}**"
            )

            st.caption(
                "The equation is generated dynamically from the final model."
            )

            insight1, insight2 = st.columns(2)

            with insight1:

                st.metric(
                    "Regression Coefficient",
                    f"{coefficient:.4f}"
                )

            with insight2:

                st.metric(
                    "Model Intercept",
                    f"{intercept:.4f}"
                )

            if coefficient > 0:

                interpretation = (
                    f"An increase of **1 Cr in "
                    f"{best_features[0].replace('_', ' ')}** "
                    f"is associated with an estimated "
                    f"**{coefficient:.4f} Cr increase** in Actual GST."
                )

            elif coefficient < 0:

                interpretation = (
                    f"An increase of **1 Cr in "
                    f"{best_features[0].replace('_', ' ')}** "
                    f"is associated with an estimated "
                    f"**{abs(coefficient):.4f} Cr decrease** in Actual GST."
                )

            else:

                interpretation = (
                    "The coefficient is approximately zero, indicating "
                    "little linear relationship with Actual GST."
                )

            st.info(
                f"📌 **Model Interpretation**\n\n{interpretation}"
            )

            coefficient_df = pd.DataFrame(
                {
                    "Feature": [
                        best_features[0].replace("_", " ")
                    ],
                    "Coefficient": [
                        coefficient
                    ]
                }
            )

            coefficient_fig = px.bar(
                coefficient_df,
                x="Feature",
                y="Coefficient",
                text="Coefficient",
                title="Linear Regression Coefficient"
            )

            coefficient_fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside"
            )

            coefficient_fig.update_layout(
                template="plotly_white",
                xaxis_title="Feature",
                yaxis_title="Coefficient",
                showlegend=False,
                height=380
            )
            font=dict(
                        size=14,
                        color="#374151"
                    ),
                    
            title_x=0,
                    
            paper_bgcolor="white",
                    
            plot_bgcolor="white"

            st.plotly_chart(
                coefficient_fig,
                use_container_width=True,
                key="prediction_coefficient_visualization"
            )

        else:

            coefficient_df = pd.DataFrame(
                {
                    "Feature": [
                        feature.replace("_", " ")
                        for feature in best_features
                    ],
                    "Coefficient": coefficients
                }
            )

            coefficient_fig = px.bar(
                coefficient_df,
                x="Feature",
                y="Coefficient",
                text="Coefficient",
                title="Selected Model Feature Coefficients"
            )

            coefficient_fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside"
            )

            coefficient_fig.update_layout(
                template="plotly_white",
                xaxis_title="Feature",
                yaxis_title="Coefficient",
                showlegend=False,
                height=380
            )
            font=dict(
                        size=14,
                        color="#374151"
                    ),
                    
            title_x=0,
                    
            paper_bgcolor="white",
                    
            plot_bgcolor="white"

            st.plotly_chart(
                coefficient_fig,
                use_container_width=True,
                key="prediction_multi_coefficient_visualization"
            )

            st.info(
                f"📌 **Model Interpretation**\n\n"
                f"The selected model uses **{len(best_features)} features**. "
                f"Positive coefficients indicate an estimated positive linear "
                f"relationship with Actual GST, while negative coefficients "
                f"indicate an estimated negative relationship."
            )

        st.divider()

        # ==================================================
        # OUTWARD TAX VS ACTUAL GST
        # ==================================================

        st.markdown("📈 Feature Relationship")

        relationship_feature = best_features[0]

        scatter_fig = px.scatter(
            train_df,
            x=relationship_feature,
            y="Actual_tax",
            trendline="ols",
            title=(
                f"{relationship_feature.replace('_', ' ')} "
                "vs Actual GST"
            )
        )

        scatter_fig.update_layout(
            template="plotly_white",
            xaxis_title=(
                f"{relationship_feature.replace('_', ' ')} (Cr)"
            ),
            yaxis_title="Actual GST (Cr)",
            height=450
        )
        font=dict(
                    size=14,
                    color="#374151"
                ),
                
        title_x=0,
                
        paper_bgcolor="white",
                
        plot_bgcolor="white"
        st.plotly_chart(
            scatter_fig,
            use_container_width=True,
            key="prediction_feature_vs_actual"
        )

        st.caption(
            "The trendline represents the linear relationship learned from "
            "the historical training data."
        )

        # ==================================================
        # ACTUAL VS PREDICTED
        # ==================================================

        st.markdown("📈 Model Validation — Actual vs Predicted")

        st.caption(
    "Comparison of actual GST revenue against model predictions "
    "for the held-out test period."
)

        evaluation_df = pd.DataFrame(
            {
                "Month": testing_data["Month"].values,
                "Actual GST": y_test.values,
                "Predicted GST": best_test_predictions
            }
        )

        evaluation_long = evaluation_df.melt(
            id_vars="Month",
            value_vars=[
                "Actual GST",
                "Predicted GST"
            ],
            var_name="Type",
            value_name="GST"
        )

        fig_test = px.line(
            evaluation_long,
            x="Month",
            y="GST",
            color="Type",
            markers=True,
            title="Actual vs Predicted GST — Test Period"
        )

        fig_test.update_layout(
            hovermode="x unified",
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="GST Revenue (Cr)",
            height=450
        )

        st.plotly_chart(
            fig_test,
            use_container_width=True,
            key="prediction_actual_vs_test"
        )

        st.divider()

        # ==================================================
# FINAL FORECAST + LATEST FORECAST EVALUATION
# ==================================================

st.markdown("🔮 GST Revenue Forecast")

st.markdown(
    """
    ### 📌 Latest GST Forecast

    The model uses tax indicators from the previous month to forecast the next month's GST collection.
    """
)

# --------------------------------------------------
# 1. LATEST ACTUAL MONTH
# --------------------------------------------------

historical_data = (
    df[df[target].notna()]
    .sort_values("Month")
    .copy()
)

future_data = (
    df[df[target].isna()]
    .dropna(subset=best_features)
    .sort_values("Month")
    .copy()
)

# --------------------------------------------------
# 2. LATEST FORECAST PERFORMANCE — INDIA
# --------------------------------------------------

# India logic:
# The model predicts the SAME month represented by the
# feature values. When that month's Actual_tax becomes
# available, we compare prediction vs actual.
#
# Example:
# July 2026 inputs → July 2026 prediction
# July 2026 Actual_tax → July 2026 actual
#
# If July Actual_tax is still blank, July remains the
# current forecast month.

if not future_data.empty:

    # Latest month that currently has no actual value
    latest_forecast_row = future_data.iloc[0]

    latest_forecast_month = latest_forecast_row["Month"]

    # --------------------------------------------------
    # CHECK WHETHER THIS FORECAST MONTH ALREADY
    # HAS AN ACTUAL VALUE
    # --------------------------------------------------

    matching_actual = historical_data[
        historical_data["Month"] == latest_forecast_month
    ]

    # --------------------------------------------------
    # IF ACTUAL IS AVAILABLE → EVALUATE FORECAST
    # --------------------------------------------------

    if not matching_actual.empty:

        actual_value = matching_actual.iloc[0][target]

        # Train ONLY on months before the month being evaluated.
        # This prevents the actual value of the evaluation month
        # from leaking into the model.
        training_before_forecast = historical_data[
            historical_data["Month"] < latest_forecast_month
        ].copy()

        if len(training_before_forecast) >= 2:

            evaluation_model = LinearRegression()

            evaluation_model.fit(
                training_before_forecast[best_features],
                training_before_forecast[target]
            )

            # Predict the evaluated month
            predicted_value = evaluation_model.predict(
                latest_forecast_row[
                    best_features
                ].to_frame().T
            )[0]

            # --------------------------------------------------
            # CALCULATE ERROR
            # --------------------------------------------------

            prediction_error = (
                abs(predicted_value - actual_value)
                / abs(actual_value)
            ) * 100 if actual_value != 0 else np.nan

            forecast_month_name = (
                latest_forecast_month.strftime("%B %Y")
            )

            # --------------------------------------------------
            # DISPLAY
            # --------------------------------------------------

            st.markdown(
                "### 🎯 Latest Forecast Performance"
            )

            st.caption(
                "The latest month's GST prediction is compared "
                "with its official actual collection."
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "📅 Month",
                forecast_month_name
            )

            p2.metric(
                "🤖 Predicted GST",
                f"{predicted_value:.2f} Cr"
            )

            p3.metric(
                "💰 Actual GST",
                f"{actual_value:.2f} Cr"
            )

            p4.metric(
                "🎯 Prediction Error",
                f"{prediction_error:.2f}%"
            )

            # --------------------------------------------------
            # FORECAST QUALITY
            # --------------------------------------------------

            if prediction_error <= 5:

                st.success(
                    f"✅ Excellent forecast! "
                    f"The model predicted {forecast_month_name} "
                    f"within {prediction_error:.2f}% of the "
                    f"actual GST collection."
                )

            elif prediction_error <= 10:

                st.info(
                    f"ℹ️ Good forecast. "
                    f"The prediction error for {forecast_month_name} "
                    f"was {prediction_error:.2f}%."
                )

            else:

                st.warning(
                    f"⚠️ The prediction error for "
                    f"{forecast_month_name} was "
                    f"{prediction_error:.2f}%."
                )

        else:

            st.info(
                "ℹ️ Not enough historical data to "
                "evaluate this forecast."
            )

    # --------------------------------------------------
    # IF ACTUAL IS NOT AVAILABLE → SHOW FORECAST STATUS
    # --------------------------------------------------

    else:

        forecast_month_name = (
            latest_forecast_month.strftime("%B %Y")
        )

        st.markdown(
            "### 🔮 Current Month Forecast"
        )

        st.caption(
            f"Official GST collection for {forecast_month_name} "
            "is not available yet."
        )

        # Train final model using all currently available
        # historical actual data.
        evaluation_model = LinearRegression()

        evaluation_model.fit(
            train_df[best_features],
            train_df[target]
        )

        current_prediction = evaluation_model.predict(
            latest_forecast_row[
                best_features
            ].to_frame().T
        )[0]

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "📅 Forecast Month",
            forecast_month_name
        )

        p2.metric(
            "🤖 Predicted GST",
            f"{current_prediction:.2f} Cr"
        )

        p3.metric(
            "📌 Status",
            "Awaiting Actual"
        )

        st.info(
            f"📊 Once the official {forecast_month_name} "
            "GST collection is entered, this forecast will "
            "automatically become a Prediction vs Actual "
            "evaluation."
        )

st.divider()


# --------------------------------------------------
# 3. FUTURE / UPCOMING MONTH FORECAST
# --------------------------------------------------

if not future_data.empty:

    # Train final model using ALL currently available actual data
    final_model = LinearRegression()

    final_model.fit(
        train_df[best_features],
        train_df[target]
    )

    # Generate predictions for all future blank months
    final_predictions = final_model.predict(
        future_data[best_features]
    )

    future_data["Predicted_tax"] = final_predictions

    st.markdown("### 🔮 Upcoming Month Forecast")

    st.caption(
        "These forecasts are generated for months where official "
        "GST collection data is not yet available."
    )

    forecast_cols = st.columns(
        len(future_data)
        if len(future_data) <= 3
        else 3
    )

    for position, (_, row) in enumerate(
        future_data.iterrows()
    ):

        with forecast_cols[
            position % len(forecast_cols)
        ]:

            st.markdown(
                f"### 📅 {row['Month'].strftime('%B %Y')}"
            )

            st.success(
                f"""
                ## 🎯 {row['Predicted_tax']:.2f} Cr

                **Predicted GST Revenue**
                """
            )

            st.metric(
                "Outward Tax",
                f"{row['Outward_tax']:.2f} Cr"
            )

            if "Inward_tax" in row:

                st.metric(
                    "Inward Tax",
                    f"{row['Inward_tax']:.2f} Cr"
                )

            if "Difference_tax" in row:

                st.metric(
                    "Difference Tax",
                    f"{row['Difference_tax']:.2f} Cr"
                )


    # --------------------------------------------------
    # 4. DOWNLOAD FORECAST
    # --------------------------------------------------

    st.divider()

    forecast_download = future_data[
        [
            "Month",
            "Outward_tax",
            "Inward_tax",
            "Difference_tax",
            "Predicted_tax"
        ]
    ].copy()

    forecast_download["Month"] = (
        forecast_download["Month"]
        .dt.strftime("%B %Y")
    )

    forecast_download = forecast_download.rename(
        columns={
            "Outward_tax": "Outward Tax (Cr)",
            "Inward_tax": "Inward Tax (Cr)",
            "Difference_tax": "Difference Tax (Cr)",
            "Predicted_tax": "Predicted GST (Cr)"
        }
    )

    forecast_csv = forecast_download.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download GST Forecast CSV",
        data=forecast_csv,
        file_name="GST_Revenue_Forecast.csv",
        mime="text/csv",
        key="download_gst_forecast"
    )


    # --------------------------------------------------
    # 5. HISTORICAL + FORECAST CHART
    # --------------------------------------------------

    st.divider()

    st.markdown(
        "### 📊 Historical Performance + Future Forecast"
    )

    st.caption(
        "Historical actual GST collections are shown together "
        "with model-generated forecasts for upcoming months."
    )

    actual_chart = df[
        ["Month", "Actual_tax"]
    ].dropna().copy()

    actual_chart["Type"] = "Actual GST"

    actual_chart = actual_chart.rename(
        columns={
            "Actual_tax": "GST"
        }
    )

    prediction_chart = future_data[
        ["Month", "Predicted_tax"]
    ].copy()

    prediction_chart["Type"] = "Forecast GST"

    prediction_chart = prediction_chart.rename(
        columns={
            "Predicted_tax": "GST"
        }
    )

    chart_df = pd.concat(
        [
            actual_chart,
            prediction_chart
        ],
        ignore_index=True
    )

    fig_forecast = px.line(
        chart_df,
        x="Month",
        y="GST",
        color="Type",
        markers=True,
        title="Historical GST Collection and Future Forecast"
    )

    fig_forecast.update_layout(
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="GST Revenue (Cr)",
        height=500
    )

    fig_forecast.update_traces(
        line_width=3,
        marker_size=8
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True,
        key="prediction_historical_forecast"
    )

else:

    st.success(
        "✅ All available months have actual GST values. "
        "Once a new month is added with a blank Actual_tax value, "
        "the model will automatically generate a forecast."
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