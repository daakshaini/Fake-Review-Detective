import streamlit as st
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake Review Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("fake_review_svm_pipeline.pkl")


model = load_model()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       PAGE
       ====================================================== */

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* ======================================================
       SIDEBAR COLLAPSE / EXPAND ARROW
       ====================================================== */

    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stExpandSidebarButton"] {
        background-color: #111827 !important;
        color: white !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="stExpandSidebarButton"]:hover {
        background-color: #1f2937 !important;
        color: white !important;
    }

    button[data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="stExpandSidebarButton"] svg {
        color: white !important;
        fill: white !important;
        stroke: white !important;
    }

    /* ======================================================
       TITLE
       ====================================================== */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 4px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 17px;
        color: #64748b;
        margin-bottom: 22px;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #172554 0%,
            #1d4ed8 100%
        );
        border-radius: 18px;
        padding: 28px 30px;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px rgba(30, 64, 175, 0.16);
    }

    .hero-title {
        color: white;
        font-size: 29px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .hero-text {
        color: #dbeafe;
        font-size: 15px;
        line-height: 1.5;
        max-width: 800px;
        margin: auto;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 14px;
        padding: 6px 14px;
        border-radius: 20px;
        background-color: rgba(255,255,255,0.14);
        color: white;
        font-size: 12px;
        font-weight: 600;
    }

    /* ======================================================
       SECTION TITLES
       ====================================================== */

    .section-title {
        color: #172554;
        font-size: 25px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 4px;
    }

    .section-text {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 12px;
    }

    /* ======================================================
       REVIEW CARD
       ====================================================== */

    .review-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    }

    /* ======================================================
       RESULT CARDS
       ====================================================== */

    .truth-result {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-top: 12px;
    }

    .truth-title {
        color: #047857;
        font-size: 27px;
        font-weight: 800;
    }

    .deceptive-result {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-top: 12px;
    }

    .deceptive-title {
        color: #c2410c;
        font-size: 27px;
        font-weight: 800;
    }

    .result-text {
        color: #475569;
        font-size: 14px;
        line-height: 1.5;
        margin-top: 7px;
    }

    /* ======================================================
       WORKFLOW
       ====================================================== */

    .workflow {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        padding: 16px 8px;
        text-align: center;
        min-height: 115px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
    }

    .workflow-icon {
        font-size: 26px;
        margin-bottom: 6px;
    }

    .workflow-name {
        color: #172554;
        font-size: 14px;
        font-weight: 700;
    }

    .workflow-text {
        color: #64748b;
        font-size: 11px;
        margin-top: 4px;
        line-height: 1.3;
    }

    /* ======================================================
       BUTTON
       ====================================================== */

    div.stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 9px;
        font-size: 15px;
        font-weight: 700;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 20px;
        padding-bottom: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LEFT SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🔍 Fake Review Detective")

    st.divider()

    st.markdown("### 📊 Project Performance")

    st.write("**Dataset:** 1,596 unique reviews")
    st.write("**Model:** Linear Support Vector Machine")
    st.write("**Features:** TF-IDF")

    st.divider()

    st.markdown("### 📈 Test Results")

    st.write("**Accuracy:** 90.00%")
    st.write("**Precision:** 89.02%")
    st.write("**Recall:** 91.25%")
    st.write("**F1-Score:** 90.12%")

    st.divider()

    st.markdown("### 🧪 Evaluation")

    st.write("Held-out test reviews: **320**")
    st.write("Correct predictions: **288**")
    st.write("Misclassified: **32**")

    st.divider()

    st.caption("Academic Data Science Project")

# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Fake Review Detective</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'NLP and Machine Learning Based Hotel Review Analysis'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">'
    'Intelligent Hotel Review Analysis'
    '</div>'
    '<div class="hero-text">'
    'Analyze hotel review text using Natural Language Processing '
    'and Machine Learning to identify linguistic patterns '
    'associated with truthful and potentially deceptive reviews.'
    '</div>'
    '<div class="hero-tag">'
    'NLP &nbsp;•&nbsp; TF-IDF &nbsp;•&nbsp; Linear SVM'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dataset",
        "1,596",
        "Unique reviews"
    )

with col2:
    st.metric(
        "Model",
        "Linear SVM",
        "Final classifier"
    )

with col3:
    st.metric(
        "Accuracy",
        "90.00%",
        "Held-out test set"
    )

with col4:
    st.metric(
        "F1-Score",
        "90.12%",
        "Held-out test set"
    )

st.divider()

# ============================================================
# REVIEW ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">📝 Review Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'Enter a hotel review below and analyze its linguistic patterns.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# REVIEW INPUT
# ============================================================

with st.container(border=True):

    st.markdown("### ✍️ Enter Hotel Review")

    sample_reviews = {
        "No sample selected": "",

        "Sample 1 — Positive Experience": (
            "The hotel was comfortable and clean. The room was spacious, "
            "the staff were polite, and the location was convenient. "
            "I enjoyed my stay and would consider returning."
        ),

        "Sample 2 — Highly Exaggerated": (
            "This hotel is absolutely perfect in every possible way! "
            "Everything was unbelievably amazing and the staff were "
            "the best I have ever seen. I cannot say enough wonderful "
            "things about this incredible hotel and everyone should "
            "stay here immediately!"
        ),

        "Sample 3 — Detailed Experience": (
            "I stayed at this hotel for three nights. The check-in "
            "process was quick, the room was clean, and the bed was "
            "comfortable. The staff answered my questions and the "
            "hotel was close to several restaurants."
        )
    }

    selected_sample = st.selectbox(
        "🧪 Try a sample review",
        list(sample_reviews.keys())
    )

    selected_text = sample_reviews[selected_sample]

    review = st.text_area(
        "Review",
        value=selected_text,
        placeholder="Paste or type your hotel review here...",
        height=210,
        label_visibility="collapsed"
    )

    analyze_button = st.button(
        "🔎 Analyze Review",
        use_container_width=True
    )

# ============================================================
# ANALYSIS RESULT
# ============================================================

if analyze_button:

    if not review.strip():

        st.warning(
            "⚠️ Please enter a hotel review before analyzing."
        )

    else:

        prediction = model.predict([review])[0]

        decision_score = float(
            model.decision_function([review])[0]
        )

        st.markdown(
            '<div class="section-title">📊 Analysis Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.markdown(
                '<div class="deceptive-result">'
                '<div class="deceptive-title">'
                '⚠️ Potentially Deceptive'
                '</div>'
                '<div class="result-text">'
                'The model detected linguistic patterns associated '
                'with deceptive reviews in its training data.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            direction = "Deceptive"

        else:

            st.markdown(
                '<div class="truth-result">'
                '<div class="truth-title">'
                '✅ Likely Truthful'
                '</div>'
                '<div class="result-text">'
                'The model detected linguistic patterns associated '
                'with truthful reviews in its training data.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            direction = "Truthful"

        score_col1, score_col2 = st.columns(2)

        with score_col1:

            st.metric(
                "SVM Decision Score",
                f"{decision_score:.4f}"
            )

        with score_col2:

            st.metric(
                "Decision Direction",
                direction
            )

        st.info(
            "ℹ️ The SVM decision score is not a probability "
            "or confidence percentage. It indicates the position "
            "of the review relative to the model's decision boundary."
        )

# ============================================================
# WORKFLOW
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">⚙️ How the System Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'The review follows the machine-learning pipeline below.'
    '</div>',
    unsafe_allow_html=True
)

w1, w2, w3, w4, w5 = st.columns(5)

workflow_data = [
    ("📝", "Review", "User enters review"),
    ("🧹", "Cleaning", "Text preprocessing"),
    ("🔤", "TF-IDF", "Text to features"),
    ("🤖", "Linear SVM", "Classification"),
    ("📊", "Prediction", "Result displayed")
]

for column, data in zip(
    [w1, w2, w3, w4, w5],
    workflow_data
):

    icon, name, description = data

    with column:

        st.markdown(
            '<div class="workflow">'
            f'<div class="workflow-icon">{icon}</div>'
            f'<div class="workflow-name">{name}</div>'
            f'<div class="workflow-text">{description}</div>'
            '</div>',
            unsafe_allow_html=True
        )

# ============================================================
# PROJECT DETAILS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📚 Project Details</div>',
    unsafe_allow_html=True
)

detail1, detail2 = st.columns(2)

with detail1:

    with st.expander("🤖 Model Evaluation"):

        st.write(
            """
            Models evaluated:

            • Logistic Regression
            • Multinomial Naive Bayes
            • Linear SVM

            Final Linear SVM:

            Accuracy: 90.00%

            Precision: 89.02%

            Recall: 91.25%

            F1-Score: 90.12%
            """
        )

with detail2:

    with st.expander("⚠️ Limitations"):

        st.write(
            """
            • The model is trained specifically on hotel reviews.

            • The system primarily uses textual information.

            • It cannot verify whether a reviewer actually stayed
              at the hotel.

            • It cannot independently verify factual claims.

            • A prediction does not prove that a review is fake
              or genuine.
            """
        )

# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ Important: This system provides a machine-learning "
    "prediction based on linguistic patterns learned from the "
    "dataset. It does not prove that a review is genuinely "
    "truthful or actually fake."
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '🔍 Fake Review Detective<br>'
    'NLP + TF-IDF + Linear SVM<br>'
    'Academic Data Science Project'
    '</div>',
    unsafe_allow_html=True
)