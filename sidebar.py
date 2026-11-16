import streamlit as st


def render_sidebar(sheet_names):

    # --------------------------------------------------
    # SIDEBAR CSS
    # --------------------------------------------------

    st.markdown(
        """
        <style>

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0f172a 0%,
                #111827 55%,
                #0b1120 100%
            );
        }

        /* Sidebar content */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* All sidebar text */
        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        /* Brand */
        .sidebar-brand {
            text-align: center;
            padding: 10px 5px 18px 5px;
        }

        .sidebar-logo {
            font-size: 38px;
            margin-bottom: 4px;
        }

        .sidebar-title {
            font-size: 23px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff !important;
        }

        .sidebar-subtitle {
            font-size: 12px;
            color: #94a3b8 !important;
            margin-top: 3px;
        }

        /* Divider */
        .sidebar-divider {
            height: 1px;
            background: rgba(148, 163, 184, 0.18);
            margin: 8px 0 20px 0;
        }

        /* Section labels */
        .sidebar-section {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b !important;
            margin-top: 18px;
            margin-bottom: 8px;
        }

        /* Dataset card */
        .dataset-card {
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
        }

        .dataset-label {
            font-size: 11px;
            color: #94a3b8 !important;
            margin-bottom: 4px;
        }

        .dataset-icon {
            font-size: 17px;
            margin-right: 5px;
        }

        /* Module card */
        .module-card {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 12px;
            padding: 12px 14px;
            margin-top: 8px;
        }

        .module-item {
            padding: 8px 4px;
            font-size: 13px;
            color: #cbd5e1 !important;
        }

        .module-item span {
            margin-right: 8px;
        }

        /* Status */
        .status-card {
            background: rgba(22, 101, 52, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 10px;
            padding: 10px 12px;
            margin-top: 20px;
            text-align: center;
        }

        .status-online {
            color: #4ade80 !important;
            font-size: 13px;
            font-weight: 700;
        }

        .status-text {
            color: #94a3b8 !important;
            font-size: 10px;
            margin-top: 3px;
        }

        /* Footer */
        .sidebar-footer {
            text-align: center;
            margin-top: 25px;
            padding-top: 15px;
            border-top: 1px solid rgba(148, 163, 184, 0.12);
        }

        .footer-main {
            font-size: 10px;
            font-weight: 600;
            color: #64748b !important;
        }

        .footer-sub {
            font-size: 9px;
            color: #475569 !important;
            margin-top: 4px;
        }

        /* Selectbox */
        [data-testid="stSidebar"] .stSelectbox label {
            display: none;
        }

        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 9px;
        }

        [data-testid="stSidebar"] .stSelectbox > div > div:hover {
            border-color: #64748b;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # BRANDING
    # --------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">📊</div>

            <div class="sidebar-title">
                GST Analytics
            </div>

            <div class="sidebar-subtitle">
                Revenue Forecasting & Tax Intelligence
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dataset-card">
            <div class="dataset-label">
                <span class="dataset-icon">📂</span>
                Select data source
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    sheet = st.selectbox(
        "Dataset",
        sheet_names,
        label_visibility="collapsed"
    )


    # --------------------------------------------------
    # DASHBOARD MODULES
    # --------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Dashboard Modules</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="module-card">

            <div class="module-item">
                <span>🏠</span> Overview
            </div>

            <div class="module-item">
                <span>📋</span> Dataset
            </div>

            <div class="module-item">
                <span>📈</span> Tax Analysis
            </div>

            <div class="module-item">
                <span>🤖</span> ML Prediction
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------

    st.markdown(
        """
        <div class="status-card">

            <div class="status-online">
                
            </div>

            <div class="status-text">
                Prediction engine ready
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-footer">

            <div class="footer-main">
                MACHINE LEARNING • TAX ANALYTICS
            </div>

            <div class="footer-sub">
                GST Revenue Intelligence Platform
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    return sheet