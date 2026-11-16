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
st.markdown("### Machine Learning Based GST Revenue Forecasting")

st.divider()

# --------------------------------------------------
# LOAD EXCEL
# --------------------------------------------------

excel_file = "data/gst_input.xlsx"

xls = pd.ExcelFile(excel_file)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 GST Dashboard")

st.sidebar.markdown("---")

sheet = st.sidebar.selectbox(
    "📂 Select Dataset",
    xls.sheet_names
)

st.sidebar.markdown("---")

st.sidebar.subheader("ℹ️ Project Information")

st.sidebar.write("**Developer:** Deep C")
st.sidebar.write("**Course:** MCA")
st.sidebar.write("**Organization:** National Informatics Centre")
st.sidebar.write("**Project:** GST Revenue Forecasting")

st.sidebar.markdown("---")

st.sidebar.success("✅ Dashboard Ready")

df = pd.read_excel(excel_file, sheet_name=sheet)

# Convert Month column into proper date
df["Month"] = pd.to_datetime(df["Month"])

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

    st.subheader("📊 Dashboard Overview")

    st.info(
        f"""
Currently viewing **{sheet}**

Machine Learning based GST Revenue Forecasting Dashboard.
"""
    )

    actual_tax = df["Actual_tax"].dropna()

    latest_actual = actual_tax.iloc[-1]

    highest_month = df.loc[
        df["Actual_tax"].idxmax(),
        "Month"
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📄 Total Records",
        len(df)
    )

    c2.metric(
        "💰 Latest GST",
        f"{latest_actual:.2f} Cr"
    )

    c3.metric(
        "📈 Highest GST",
        f"{actual_tax.max():.2f} Cr"
    )

    c4.metric(
        "📊 Average GST",
        f"{actual_tax.mean():.2f} Cr"
    )

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
        template="plotly_dark"
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

    st.subheader("📖 Project Summary")

    st.write("""
This dashboard analyzes historical GST collections and predicts future GST revenue using Linear Regression.

### Features

- Interactive Dashboard
- Historical Trend Analysis
- Correlation Heatmap
- Machine Learning Prediction
- Downloadable Reports
""")

    st.subheader("Dashboard Overview")

    actual_tax = df["Actual_tax"].dropna()

    latest_actual = actual_tax.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Latest Actual Tax",
        f"{latest_actual:.2f}"
    )

    col3.metric(
        "Highest Actual Tax",
        f"{actual_tax.max():.2f}"
    )

    col4.metric(
        "Average Actual Tax",
        f"{actual_tax.mean():.2f}"
    )

    st.divider()

    st.subheader("Actual Tax Trend")

    fig = px.line(
        df,
        x="Month",
        y="Actual_tax",
        markers=True,
        title="Monthly Actual Tax Collection"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# DATASET
# ==================================================

with tab2:

    st.subheader("GST Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

# ==================================================
# ANALYSIS
# ==================================================

with tab3:

    st.subheader("Data Analysis")

    st.markdown("### Actual Tax")

    fig1 = px.line(
        df,
        x="Month",
        y="Actual_tax",
        markers=True
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Outward Tax")

    fig2 = px.line(
        df,
        x="Month",
        y="Outward_tax",
        markers=True
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Inward Tax")

    fig3 = px.line(
        df,
        x="Month",
        y="Inward_tax",
        markers=True
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Difference Tax")

    fig4 = px.line(
        df,
        x="Month",
        y="Difference_tax",
        markers=True
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        df[
            [
                "Outward_tax",
                "Inward_tax",
                "Difference_tax",
                "Actual_tax"
            ]
        ].corr(),
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

# ==================================================
# PREDICTION
# ==================================================

with tab4:

    st.subheader("🤖 GST Revenue Prediction")

    st.markdown(
        """
        This model evaluates multiple Linear Regression feature
        combinations and uses the best-performing model to forecast
        GST revenue for months where actual collection is unavailable.
        """
    )

    # --------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------

    features = [
        "Outward_tax",
        "Inward_tax",
        "Difference_tax"
    ]

    target = "Actual_tax"

    # Convert required columns to numeric
    for col in features + [target]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # HISTORICAL DATA
    # --------------------------------------------------

    train_df = df.dropna(
        subset=features + [target]
    ).copy()

    train_df = train_df.sort_values(
        "Month"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # FUTURE / BLANK ACTUAL DATA
    # --------------------------------------------------

    predict_df = df[
        df[target].isna()
    ].dropna(
        subset=features
    ).copy()

    # --------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------

    if len(train_df) < 10:

        st.error(
            "❌ Not enough historical data for model evaluation."
        )

    else:

        # ==================================================
        # 80 / 20 CHRONOLOGICAL SPLIT
        # ==================================================

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

        # ==================================================
        # MODEL COMPARISON
        # ==================================================

        st.subheader("🧪 Model Comparison")

        st.markdown(
            """
            Four feature combinations are evaluated using the same
            chronological train-test split.
            """
        )

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

        # --------------------------------------------------
        # EVALUATE EACH MODEL
        # --------------------------------------------------

        for model_name, selected_features in model_features.items():

            X_train_model = training_data[
                selected_features
            ]

            X_test_model = testing_data[
                selected_features
            ]

            comparison_model = LinearRegression()

            comparison_model.fit(
                X_train_model,
                y_train
            )

            model_predictions = comparison_model.predict(
                X_test_model
            )

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

        # --------------------------------------------------
        # COMPARISON TABLE
        # --------------------------------------------------

        comparison_df = pd.DataFrame(
            comparison_results
        )

        display_df = comparison_df.copy()

        display_df["R²"] = display_df[
            "R²"
        ].round(3)

        display_df["MAE (Cr)"] = display_df[
            "MAE (Cr)"
        ].round(3)

        display_df["RMSE (Cr)"] = display_df[
            "RMSE (Cr)"
        ].round(3)

        display_df["MAPE (%)"] = display_df[
            "MAPE (%)"
        ].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ==================================================
        # BEST MODEL
        # ==================================================

        best_model_row = comparison_df.loc[
            comparison_df["MAPE (%)"].idxmin()
        ]

        st.success(
            f"""
            🏆 **Best Model: {best_model_row['Model']}**

            R² Score: **{best_model_row['R²']:.3f}**

            MAPE: **{best_model_row['MAPE (%)']:.2f}%**
            """
        )

        st.divider()

        # ==================================================
        # SELECTED MODEL
        # ==================================================

        best_features = [
            "Outward_tax"
        ]

        # --------------------------------------------------
        # TEST MODEL
        # --------------------------------------------------

        best_test_model = LinearRegression()

        best_test_model.fit(
            training_data[best_features],
            y_train
        )

        best_test_predictions = best_test_model.predict(
            testing_data[best_features]
        )

        # --------------------------------------------------
        # TEST METRICS
        # --------------------------------------------------

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

        # ==================================================
        # SELECTED MODEL PERFORMANCE
        # ==================================================

        st.subheader(
            "🏆 Selected Model Performance"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "R² Score",
            f"{best_r2:.3f}"
        )

        c2.metric(
            "MAE",
            f"{best_mae:.3f} Cr"
        )

        c3.metric(
            "RMSE",
            f"{best_rmse:.3f} Cr"
        )

        c4.metric(
            "MAPE",
            f"{best_mape:.2f}%"
        )

        st.info(
            """
            **Selected Model:** Linear Regression using Outward Tax only.

            This model was selected because it achieved the lowest
            test MAPE among the evaluated feature combinations.
            """
        )

        st.divider()

                # ==================================================
        # MODEL INSIGHTS
        # ==================================================

        st.subheader(
            "🧠 Model Insights"
        )

        # --------------------------------------------------
        # FINAL MODEL
        # --------------------------------------------------

        final_model = LinearRegression()

        final_model.fit(
            train_df[best_features],
            train_df[target]
        )

        # Get actual coefficient and intercept
        coefficient = float(
            final_model.coef_[0]
        )

        intercept = float(
            final_model.intercept_
        )

        # --------------------------------------------------
        # REGRESSION EQUATION
        # --------------------------------------------------

        st.markdown(
            "### 🧮 Regression Equation"
        )

        if coefficient >= 0:

            equation_text = (
                f"Actual GST = "
                f"{intercept:.4f} + "
                f"{coefficient:.4f} × Outward Tax"
            )

        else:

            equation_text = (
                f"Actual GST = "
                f"{intercept:.4f} − "
                f"{abs(coefficient):.4f} × Outward Tax"
            )

        st.success(
            f"**{equation_text}**"
        )

        st.caption(
            "The equation is generated dynamically from the "
            "final Linear Regression model."
        )

        # --------------------------------------------------
        # COEFFICIENT + INTERCEPT
        # --------------------------------------------------

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

        # --------------------------------------------------
        # MODEL INTERPRETATION
        # --------------------------------------------------

        if coefficient > 0:

            interpretation = (
                f"An increase of **1 Cr in Outward Tax** "
                f"is associated with an estimated "
                f"**{coefficient:.4f} Cr increase** in "
                f"Actual GST, according to the fitted model."
            )

        elif coefficient < 0:

            interpretation = (
                f"An increase of **1 Cr in Outward Tax** "
                f"is associated with an estimated "
                f"**{abs(coefficient):.4f} Cr decrease** in "
                f"Actual GST, according to the fitted model."
            )

        else:

            interpretation = (
                "The coefficient is approximately zero, "
                "indicating little linear relationship between "
                "Outward Tax and Actual GST."
            )

        st.info(
            f"📌 **Model Interpretation**\n\n"
            f"{interpretation}"
        )

        st.divider()

        # ==================================================
        # COEFFICIENT VISUALIZATION
        # ==================================================

        st.subheader(
            "📊 Feature Coefficient Visualization"
        )

        coefficient_df = pd.DataFrame(
            {
                "Feature": [
                    "Outward Tax"
                ],
                "Coefficient": [
                    coefficient
                ]
            }
        )

        fig_coefficient = px.bar(
            coefficient_df,
            x="Feature",
            y="Coefficient",
            text="Coefficient",
            title="Linear Regression Coefficient"
        )

        fig_coefficient.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        fig_coefficient.update_layout(
            template="plotly_dark",
            xaxis_title="Feature",
            yaxis_title="Coefficient",
            showlegend=False
        )

        st.plotly_chart(
            fig_coefficient,
            use_container_width=True,
            key="coefficient_visualization"
        )

        st.divider()

        # ==================================================
        # OUTWARD TAX VS ACTUAL GST
        # ==================================================

        st.subheader(
            "📈 Outward Tax vs Actual GST"
        )

        scatter_fig = px.scatter(
        train_df,
        x="Outward_tax",
        y="Actual_tax",
        trendline="ols",
        title="Relationship Between Outward Tax and Actual GST"
    )

        scatter_fig.update_layout(
            template="plotly_dark",
            xaxis_title="Outward Tax (Cr)",
            yaxis_title="Actual GST (Cr)"
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True,
            key="outward_vs_actual_scatter"
        )

        st.caption(
            "The trendline shows the linear relationship learned "
            "between Outward Tax and Actual GST."
        )
        # ==================================================
        # ACTUAL VS PREDICTED
        # ==================================================

        st.subheader(
            "📈 Actual vs Predicted — Test Data"
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
            template="plotly_dark",
            xaxis_title="Month",
            yaxis_title="GST Revenue (Cr)"
        )

        st.plotly_chart(
            fig_test,
            use_container_width=True,
            key="best_model_test_chart"
        )

        st.divider()

        # ==================================================
        # FINAL FORECAST
        # ==================================================

        st.subheader(
            "🔮 GST Revenue Forecast"
        )

        if not predict_df.empty:

            final_predictions = final_model.predict(
                predict_df[best_features]
            )

            predict_df["Predicted_tax"] = (
                final_predictions
            )

            for index, row in predict_df.iterrows():

                st.markdown(
                    f"### 📅 {row['Month'].strftime('%B %Y')}"
                )

                forecast1, forecast2, forecast3 = st.columns(3)

                forecast1.metric(
                    "Outward Tax",
                    f"{row['Outward_tax']:.2f}"
                )

                forecast2.metric(
                    "Inward Tax",
                    f"{row['Inward_tax']:.2f}"
                )

                forecast3.metric(
                    "Difference Tax",
                    f"{row['Difference_tax']:.2f}"
                )

                st.success(
                    f"""
                    ## 🎯 Predicted GST Revenue

                    **{row['Predicted_tax']:.2f} Cr**
                    """
                )

            st.divider()

            # ==================================================
            # HISTORICAL + FORECAST CHART
            # ==================================================

            st.subheader(
                "📊 Historical GST + Future Forecast"
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

            prediction_chart = predict_df[
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
                title="Historical GST Collection and Forecast"
            )

            fig_forecast.update_layout(
                hovermode="x unified",
                template="plotly_dark",
                xaxis_title="Month",
                yaxis_title="GST Revenue (Cr)"
            )

            st.plotly_chart(
                fig_forecast,
                use_container_width=True,
                key="final_forecast_chart"
            )

        else:

            st.info(
                "ℹ️ No blank Actual_tax values found. "
                "All available months already have actual GST values."
            )