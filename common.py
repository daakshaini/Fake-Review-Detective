import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("fake_review_svm_pipeline.pkl")

def apply_style():
    st.markdown(
        """
        <style>
        .stApp { background-color: #f5f7fb; }
        .block-container {
        max-width: 1150px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
        section[data-testid="stSidebar"] { background-color: #111827; }
        section[data-testid="stSidebar"] * { color: white; }
        button[data-testid="stSidebarCollapseButton"],
        button[data-testid="stExpandSidebarButton"] {
            background-color: #111827 !important; color: white !important;
            border: 1px solid #374151 !important; border-radius: 8px !important;
        }
        button[data-testid="stSidebarCollapseButton"]:hover,
        button[data-testid="stExpandSidebarButton"]:hover {
            background-color: #1f2937 !important; color: white !important;
        }
        button[data-testid="stSidebarCollapseButton"] svg,
        button[data-testid="stExpandSidebarButton"] svg {
            color: white !important; fill: white !important; stroke: white !important;
        }
        .main-title { text-align: center; font-size: 42px; font-weight: 800; color: #172554; margin-bottom: 4px; }
        .main-subtitle { text-align: center; font-size: 17px; color: #64748b; margin-bottom: 22px; }
        .hero {
            background: linear-gradient(135deg, #172554 0%, #1d4ed8 100%);
            border-radius: 18px; padding: 28px 30px; text-align: center; margin-bottom: 22px;
            box-shadow: 0 10px 25px rgba(30, 64, 175, 0.16);
        }
        .hero-title { color: white; font-size: 29px; font-weight: 800; margin-bottom: 7px; }
        .hero-text { color: #dbeafe; font-size: 15px; line-height: 1.5; max-width: 800px; margin: auto; }
        .hero-tag { display: inline-block; margin-top: 14px; padding: 6px 14px; border-radius: 20px; background-color: rgba(255,255,255,0.14); color: white; font-size: 12px; font-weight: 600; }
        .section-title { color: #172554; font-size: 25px; font-weight: 750; margin-top: 15px; margin-bottom: 4px; }
        .section-text { color: #64748b; font-size: 14px; margin-bottom: 12px; }
        .truth-result { background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 16px; padding: 22px; text-align: center; margin-top: 12px; }
        .truth-title { color: #047857; font-size: 27px; font-weight: 800; }
        .deceptive-result { background: #fff7ed; border: 1px solid #fed7aa; border-radius: 16px; padding: 22px; text-align: center; margin-top: 12px; }
        .deceptive-title { color: #c2410c; font-size: 27px; font-weight: 800; }
        .result-text { color: #475569; font-size: 14px; line-height: 1.5; margin-top: 7px; }
        .workflow { background: white; border: 1px solid #e2e8f0; border-radius: 13px; padding: 16px 8px; text-align: center; min-height: 115px; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04); }
        .workflow-icon { font-size: 26px; margin-bottom: 6px; }
        .workflow-name { color: #172554; font-size: 14px; font-weight: 700; }
        .workflow-text { color: #64748b; font-size: 11px; margin-top: 4px; line-height: 1.3; }
        div.stButton > button { width: 100%; min-height: 46px; border-radius: 9px; font-size: 15px; font-weight: 700; }
        .footer { text-align: center; color: #64748b; font-size: 12px; line-height: 1.5; margin-top: 20px; padding-bottom: 5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
