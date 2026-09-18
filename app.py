import streamlit as st
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fake Review Detective",
    page_icon="🔍",
    layout="centered"
)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fake_review_svm_pipeline.pkl")


model = load_model()

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .footer {
        text-align: center;
        margin-top: 35px;
        font-size: 14px;
    }

    .score-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🔍 Fake Review Detective</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP & Machine Learning Based Hotel Review Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Analyze a hotel review using a machine-learning model trained "
    "to identify linguistic patterns associated with truthful and "
    "potentially deceptive reviews."
)

st.divider()

# --------------------------------------------------
# Project Information
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", "1,596 Reviews")

with col2:
    st.metric("Best Model", "Linear SVM")

with col3:
    st.metric("Test Accuracy", "90.00%")

st.divider()

# --------------------------------------------------
# Sample Reviews
# --------------------------------------------------
st.subheader("🧪 Try a Sample Review")

sample_reviews = {
    "Select a sample": "",
    "Sample 1 — Positive": (
        "The hotel was comfortable and clean. The room was spacious, "
        "the staff were polite, and the location was convenient. "
        "I enjoyed my stay and would consider returning."
    ),
    "Sample 2 — Exaggerated": (
        "This hotel is absolutely perfect in every possible way! "
        "Everything was unbelievably amazing and the staff were the "
        "best I have ever seen. I cannot say enough wonderful things "
        "about this incredible hotel and everyone should stay here "
        "immediately!"
    ),
    "Sample 3 — Detailed Experience": (
        "I stayed at this hotel for three nights. The check-in process "
        "was quick, the room was clean, and the bed was comfortable. "
        "The staff answered my questions and the hotel was close to "
        "several restaurants."
    )
}

selected_sample = st.selectbox(
    "Choose a sample review:",
    list(sample_reviews.keys())
)

selected_text = sample_reviews[selected_sample]

# --------------------------------------------------
# Review Input
# --------------------------------------------------
st.subheader("📝 Enter Hotel Review")

review = st.text_area(
    "Paste or type the hotel review below:",
    value=selected_text,
    placeholder=(
        "Example: The room was clean and comfortable. "
        "The staff were friendly and the location was excellent..."
    ),
    height=220
)

# --------------------------------------------------
# Analyze Button
# --------------------------------------------------
if st.button("🔎 Analyze Review", use_container_width=True):

    if not review.strip():

        st.warning("⚠️ Please enter a hotel review before analyzing.")

    else:

        # Prediction
        prediction = model.predict([review])[0]

        # SVM decision score
        decision_score = float(model.decision_function([review])[0])

        st.divider()
        st.subheader("📊 Analysis Result")

        # --------------------------------------------------
        # Prediction Result
        # --------------------------------------------------
        if prediction == 1:

            st.error("⚠️ Potentially Deceptive")

            st.write(
                "The model detected linguistic patterns that are "
                "associated with deceptive reviews in its training data."
            )

        else:

            st.success("✅ Likely Truthful")

            st.write(
                "The model detected linguistic patterns that are "
                "associated with truthful reviews in its training data."
            )

        # --------------------------------------------------
        # Decision Score
        # --------------------------------------------------
        st.markdown(
            '<div class="score-box">',
            unsafe_allow_html=True
        )

        st.write("### 📈 Model Decision Score")

        st.metric(
            label="SVM Decision Score",
            value=f"{decision_score:.4f}"
        )

        if decision_score > 0:
            st.write(
                "The score is on the side of the **deceptive** class."
            )
        else:
            st.write(
                "The score is on the side of the **truthful** class."
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # Disclaimer
        # --------------------------------------------------
        st.info(
            "ℹ️ Important: The decision score is not a probability "
            "or a percentage. It represents the position of the review "
            "relative to the SVM decision boundary. This system provides "
            "a machine-learning prediction based on textual patterns and "
            "does not prove whether a review is genuinely truthful or fake."
        )

# --------------------------------------------------
# About the Model
# --------------------------------------------------
st.divider()

with st.expander("🔬 About this system"):

    st.write(
        """
        **Fake Review Detective** is a Natural Language Processing (NLP)
        and Machine Learning based system designed for hotel review
        deception analysis.

        **Machine Learning Pipeline:**

        Hotel Review → Text Cleaning → TF-IDF → Linear SVM → Prediction

        **Dataset:** 1,596 unique hotel reviews

        **Final Model:** Linear Support Vector Machine (SVM)

        **Test Accuracy:** 90.00%

        **Test Precision:** 89.02%

        **Test Recall:** 91.25%

        **Test F1-Score:** 90.12%
        """
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="footer">'
    'Fake Review Detective • NLP + TF-IDF + Linear SVM'
    '</div>',
    unsafe_allow_html=True
)