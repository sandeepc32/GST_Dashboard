import streamlit as st


def load_css():
    st.markdown("""
    <style>

    /* ==========================================================
                            GOOGLE FONT
    ========================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]{
        font-family: 'Inter', sans-serif;
    }

    /* ==========================================================
                            MAIN APP
    ========================================================== */

    .block-container{
        padding-top:1.2rem;
        padding-left:2rem;
        padding-right:2rem;
        padding-bottom:2rem;
        max-width:96%;
    }

    .main{
        background-color:#0E1117;
    }

    /* ==========================================================
                            HEADINGS
    ========================================================== */

    h1{
        font-size:2.2rem;
        font-weight:700;
        color:white;
        letter-spacing:-0.5px;
    }

    h2{
        font-size:1.6rem;
        font-weight:600;
        color:white;
    }

    h3{
        font-size:1.25rem;
        font-weight:600;
        color:white;
    }

    p{
        color:#C9D1D9;
    }

    /* ==========================================================
                            SIDEBAR
    ========================================================== */

    section[data-testid="stSidebar"]{
        background:#111827;
        border-right:1px solid #2A2F3A;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3{
        color:white;
    }

    section[data-testid="stSidebar"] .stSelectbox{
        margin-bottom:15px;
    }

    /* ==========================================================
                            TABS
    ========================================================== */

    button[data-baseweb="tab"]{
        background:transparent;
        color:#9CA3AF;
        font-weight:600;
        font-size:15px;
        border-radius:10px;
        padding:12px 18px;
        transition:all .25s ease;
    }

    button[data-baseweb="tab"]:hover{
        color:white;
        background:#1F2937;
    }

    button[aria-selected="true"]{
        background:#2563EB !important;
        color:white !important;
    }

    /* ==========================================================
                            METRIC CARDS
    ========================================================== */

    div[data-testid="stMetric"]{

        background:#161B22;

        border:1px solid #30363D;

        border-radius:16px;

        padding:18px;

        transition:0.25s;

        box-shadow:0px 3px 12px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetric"]:hover{

        transform:translateY(-3px);

        border-color:#2563EB;

        box-shadow:0px 8px 24px rgba(37,99,235,.35);

    }

    div[data-testid="stMetricLabel"]{

        color:#9CA3AF;

        font-weight:600;

        font-size:15px;
    }

    div[data-testid="stMetricValue"]{

        font-size:30px;

        font-weight:700;

        color:white;
    }

    /* ==========================================================
                            BUTTONS
    ========================================================== */

    .stButton>button,
    .stDownloadButton>button{

        width:100%;

        border-radius:10px;

        border:none;

        background:#2563EB;

        color:white;

        font-weight:600;

        transition:.25s;
    }

    .stButton>button:hover,
    .stDownloadButton>button:hover{

        background:#1D4ED8;

        transform:scale(1.02);

    }

    /* ==========================================================
                            ALERTS
    ========================================================== */

    div[data-testid="stAlert"]{

        border-radius:14px;

        border:1px solid #30363D;

    }

    /* ==========================================================
                            DATAFRAME
    ========================================================== */

    div[data-testid="stDataFrame"]{

        border-radius:15px;

        overflow:hidden;

        border:1px solid #30363D;

    }

    /* ==========================================================
                            PLOTLY
    ========================================================== */

    div[data-testid="stPlotlyChart"]{

        background:#161B22;

        border:1px solid #30363D;

        border-radius:16px;

        padding:12px;

        box-shadow:0px 4px 14px rgba(0,0,0,.22);

    }

    /* ==========================================================
                            EXPANDER
    ========================================================== */

    details{

        border-radius:14px;

        border:1px solid #30363D;

        padding:8px;

    }

    /* ==========================================================
                            HORIZONTAL LINE
    ========================================================== */

    hr{

        border-color:#30363D;

        margin-top:1rem;

        margin-bottom:1rem;

    }

    /* ==========================================================
                            SCROLLBAR
    ========================================================== */

    ::-webkit-scrollbar{

        width:8px;

        height:8px;

    }

    ::-webkit-scrollbar-thumb{

        background:#374151;

        border-radius:10px;

    }

    ::-webkit-scrollbar-thumb:hover{

        background:#4B5563;

    }

    /* ==========================================================
                            FOOTER
    ========================================================== */

    footer{

        visibility:hidden;

    }

    </style>
    """, unsafe_allow_html=True)
    