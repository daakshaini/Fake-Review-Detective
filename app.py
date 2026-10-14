import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Page setup and model loading
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fake Review Detective",
    page_icon="🔍",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "fake_review_svm_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(
        "Could not load fake_review_svm_pipeline.pkl. "
        "Keep the model file in the same folder as app.py."
    )
    st.exception(exc)
    st.stop()


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4f7ff 0%, #edf3ff 50%, #f8fafc 100%);
        color: #172554;
    }
    .block-container {
    max-width: 1280px;
    padding-top: 3rem;
    padding-bottom: 2.5rem;
}

    /* Hide Streamlit's automatic page list; use our custom sidebar navigation. */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101c43 0%, #172554 55%, #1e3a8a 100%);
        border-right: 1px solid #253b72;
    }
    section[data-testid="stSidebar"] * {
        color: #eaf1ff;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 10px;
        padding: 7px 10px;
        margin: 4px 0;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(96,165,250,.18);
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.18);
    }

    .brand-heading {
        display: block;
        color: #1769e8;
        font-size: clamp(1.6rem, 2.5vw, 2.35rem);
        font-weight: 900;
        letter-spacing: 5px;
        line-height: 1.2;
        margin: 0 0 20px;
        padding: 0 0 12px;
        text-transform: uppercase;
        border-bottom: 3px solid #b9ddff;
        text-shadow: 0 2px 10px rgba(37, 99, 235, 0.08);
    }
    .hero {
        background:
            linear-gradient(115deg, rgba(10,24,67,.97), rgba(29,78,216,.90)),
            url("https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=1800&q=85")
            center/cover;
        border-radius: 26px;
        padding: 44px 40px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 20px 50px rgba(30,64,175,.22);
        border: 1px solid rgba(255,255,255,.25);
        min-height: 290px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hero .eyebrow {
        display: inline-block;
        font-size: .78rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 800;
        color: #bfdbfe;
        margin-bottom: 14px;
    }
    .hero h1 {
        color: white;
        font-size: clamp(2.2rem, 4vw, 3.25rem);
        font-weight: 850;
        letter-spacing: -1.2px;
        line-height: 1.08;
        margin: 0 0 14px;
    }
    .hero p {
        color: #eaf2ff;
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 760px;
        margin: 0;
    }
    .section-kicker {
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-weight: 800;
        font-size: .75rem;
        margin: 6px 0;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #14234a;
        letter-spacing: -.5px;
        margin: 0 0 6px;
    }
    .section-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin: 0 0 20px;
    }
    .feature-card {
        background: rgba(255,255,255,.96);
        border: 1px solid #dce7fb;
        border-radius: 18px;
        padding: 22px 22px 20px;
        min-height: 205px;
        box-shadow: 0 8px 26px rgba(30,64,175,.07);
    }
    .feature-card .icon {
        font-size: 1.7rem;
        margin-bottom: 14px;
    }
    .feature-card h3 {
        margin: 0 0 9px;
        color: #172554;
        font-size: 1.17rem;
        font-weight: 800;
    }
    .feature-card p {
        color: #64748b;
        line-height: 1.65;
        margin: 0;
    }
    .feature-tag {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: .73rem;
        font-weight: 750;
        margin-bottom: 12px;
    }
    .photo-caption {
        font-size: .85rem;
        color: #64748b;
        margin-top: -4px;
        margin-bottom: 12px;
    }
    .workflow-strip {
        background: linear-gradient(100deg, #eaf1ff, #f4f7ff);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 20px 22px;
        margin-top: 22px;
    }
    .workflow-strip h3 {
        color: #172554;
        margin: 0 0 7px;
        font-size: 1.1rem;
    }
    .workflow-strip p {
        color: #475569;
        line-height: 1.65;
        margin: 0;
    }
    .stButton button {
        border-radius: 10px;
        font-weight: 750;
        border: 0;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1d4ed8, #4f46e5);
        color: white;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #dbeafe;
        padding: 14px;
        border-radius: 14px;
        box-shadow: 0 5px 15px rgba(30,64,175,.05);
    }
    .result {
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-top: 12px;
        box-shadow: 0 8px 20px rgba(15,23,42,.06);
    }
    .truth {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
    }
    .deceptive {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #c2410c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Custom sidebar navigation
# ------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Fake Review Detective")
    page = st.radio(
        "Navigate",
        ["Home", "Single Review", "Bulk Review Analysis", "Model Performance"],
    )
    st.divider()
    st.caption("Academic Data Science Project")


# ------------------------------------------------------------
# Prediction and explanation helper
# ------------------------------------------------------------
def show_prediction(review, hotel="Not specified"):
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
        return

    prediction = int(model.predict([review])[0])

    try:
        score = float(model.decision_function([review])[0])
    except Exception:
        score = None

    st.write(f"**Hotel selected:** {hotel}")

    if prediction == 1:
        st.markdown(
            """
            <div class="result deceptive">
                <h2>⚠️ Potentially Deceptive</h2>
                <p>The model found text patterns associated with deceptive reviews in its training data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        direction = "Deceptive"
    else:
        st.markdown(
            """
            <div class="result truth">
                <h2>✅ Likely Truthful</h2>
                <p>The model found text patterns associated with truthful reviews in its training data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        direction = "Truthful"

    col1, col2 = st.columns(2)
    with col1:
        if score is not None:
            st.metric("SVM Decision Score", f"{score:.4f}")
    with col2:
        st.metric("Decision Direction", direction)

    st.info(
        "The SVM decision score is not a probability or confidence percentage. "
        "A prediction does not prove whether a review is genuine or fake."
    )

    with st.expander("Why did the model make this prediction?"):
        try:
            tfidf = model.named_steps["tfidf"]
            svm = model.named_steps["model"]
            vector = tfidf.transform([review])
            names = tfidf.get_feature_names_out()
            coefficients = svm.coef_[0]
            contributions = vector.multiply(coefficients).toarray()[0]
            indices = vector.indices

            positive = sorted(
                [
                    (names[i], float(contributions[i]))
                    for i in indices
                    if contributions[i] > 1e-8
                ],
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            negative = sorted(
                [
                    (names[i], float(contributions[i]))
                    for i in indices
                    if contributions[i] < -1e-8
                ],
                key=lambda item: item[1],
            )[:5]

            support, oppose = (
                (positive, negative) if prediction == 1 else (negative, positive)
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**Patterns supporting prediction**")
                if support:
                    for term, value in support:
                        st.write(f"• `{term}` — contribution: `{abs(value):.4f}`")
                else:
                    st.write("No strong individual word contribution found.")

            with right:
                st.markdown("**Patterns pushing the other way**")
                if oppose:
                    for term, value in oppose:
                        st.write(f"• `{term}` — contribution: `{abs(value):.4f}`")
                else:
                    st.write("No strong opposing word contribution found.")

            st.caption(
                "Feature contributions describe how text features affect the linear "
                "SVM score; they are not proof of deception."
            )
        except (AttributeError, KeyError, ValueError, IndexError):
            st.caption("Feature explanations are not available for this model file.")


# ------------------------------------------------------------
# Home
# ------------------------------------------------------------
if page == "Home":
    st.markdown(
        '<div class="brand-heading">🔍 FAKE REVIEW DETECTIVE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Hotel Review Intelligence · NLP + Machine Learning</div>
            <h1>Look beyond the stars.<br>Read between the reviews.</h1>
            <p>
                Explore hotel-review language with a TF-IDF and Linear SVM model.
                Screen individual reviews or analyze a complete CSV to spot patterns
                associated with potentially deceptive feedback.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-kicker">Your review intelligence workspace</div>
        <div class="section-title">Choose how you want to investigate</div>
        <div class="section-subtitle">
            Three focused tools, one streamlined workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3, gap="large")
    with col_a:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-tag">01 · INDIVIDUAL CHECK</div>
                <div class="icon">📝</div>
                <h3>Single Review Analysis</h3>
                <p>
                    Inspect one hotel review, see the model prediction and SVM
                    decision score, and explore the text patterns that influenced
                    the result.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-tag">02 · BATCH SCREENING</div>
                <div class="icon">📂</div>
                <h3>Bulk Review Analysis</h3>
                <p>
                    Upload review data in CSV format, screen up to 5,000 rows,
                    compare predictions, and export a results file for further analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-tag">03 · MODEL EVALUATION</div>
                <div class="icon">📊</div>
                <h3>Model Performance</h3>
                <p>
                    Review the recorded model comparison, test-set confusion matrix,
                    and cross-validation metrics used to evaluate this project.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-kicker" style="margin-top:30px">
            A closer look at hospitality
        </div>
        <div class="section-title">Hotel experiences are told in the details</div>
        <div class="section-subtitle">
            Review language can reveal patterns worth examining—but a model
            prediction is a screening signal, not proof.
        </div>
        """,
        unsafe_allow_html=True,
    )

    photo1, photo2, photo3 = st.columns(3, gap="medium")
    with photo1:
        st.image(
            "https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=900&q=85",
            use_container_width=True,
        )
        st.markdown("**The stay**")
        st.markdown(
            '<div class="photo-caption">Rooms, amenities, and the overall guest experience.</div>',
            unsafe_allow_html=True,
        )
    with photo2:
        st.image(
            "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=900&q=85",
            use_container_width=True,
        )
        st.markdown("**The service**")
        st.markdown(
            '<div class="photo-caption">Guest descriptions of staff, check-in, and hospitality.</div>',
            unsafe_allow_html=True,
        )
    with photo3:
        st.image(
            "https://images.unsplash.com/photo-1571896349842-33c89424de2d?auto=format&fit=crop&w=900&q=85",
            use_container_width=True,
        )
        st.markdown("**The details**")
        st.markdown(
            '<div class="photo-caption">Specific experiences that make reviews informative.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="workflow-strip">
            <h3>How the analysis works</h3>
            <p>
                <b>Review text</b> → TF-IDF text features → Linear SVM classification
                → prediction and decision score. The system identifies language
                patterns learned from labelled data; it does not independently
                verify whether a review is genuine.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Single review
# ------------------------------------------------------------
elif page == "Single Review":
    st.title("📝 Single Review Analysis")
    st.write(
        "Choose a hotel and enter a review. Hotel name is context only; "
        "the classifier predicts from the review text."
    )

    hotels = [
        "Affinia", "Allegro", "Amalfi", "Ambassador", "Conrad", "Fairmont",
        "Hard Rock", "Hilton", "Homewood Suites", "Hyatt", "InterContinental",
        "James", "Knickerbocker", "Monaco", "Omni", "Palmer House", "Sheraton",
        "Sofitel", "Swissotel", "Talbott",
    ]
    hotel = st.selectbox("Select hotel", hotels)

    examples = {
        "No sample selected": "",
        "Sample 1 — Positive Experience": (
            "The hotel was comfortable and clean. The room was spacious, "
            "the staff were polite, and the location was convenient. "
            "I enjoyed my stay and would consider returning."
        ),
        "Sample 2 — Highly Exaggerated": (
            "This hotel is absolutely perfect in every possible way! "
            "Everything was unbelievably amazing and the staff were the best "
            "I have ever seen. Everyone should stay here immediately!"
        ),
        "Sample 3 — Detailed Experience": (
            "I stayed at this hotel for three nights. Check-in was quick, "
            "the room was clean, and the bed was comfortable. "
            "The hotel was close to several restaurants."
        ),
    }
    sample = st.selectbox("Try a sample review", list(examples.keys()))
    review = st.text_area(
        "Hotel review",
        value=examples[sample],
        height=200,
        placeholder="Paste or type a hotel review here...",
    )

    if st.button("🔎 Analyze Review", type="primary", use_container_width=True):
        show_prediction(review, hotel)


# ------------------------------------------------------------
# Bulk review analysis
# ------------------------------------------------------------
elif page == "Bulk Review Analysis":
    st.title("📂 Bulk Review Analysis")
    st.write(
        "Upload a CSV file containing multiple reviews. "
        "The file should have one review per row."
    )

    template = pd.DataFrame(
        {
            "hotel": ["Hilton", "Conrad", "Hyatt"],
            "review": [
                "The room was clean and check-in was quick.",
                "The bathroom was dirty and the air conditioner did not work.",
                "I stayed two nights; the room was comfortable but Wi-Fi was slow.",
            ],
        }
    )
    st.download_button(
        "⬇️ Download sample CSV template",
        template.to_csv(index=False).encode("utf-8-sig"),
        "fake_review_detective_template.csv",
        "text/csv",
    )

    upload = st.file_uploader("Choose CSV file", type=["csv"])

    if upload is not None:
        try:
            df = pd.read_csv(io.BytesIO(upload.getvalue()))

            if df.empty:
                st.warning("The CSV contains no data rows.")
            elif len(df) > 5000:
                st.warning("For this demo, upload no more than 5,000 rows at a time.")
            else:
                st.caption(f"Loaded {len(df):,} rows")
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)

                normalized = {
                    str(column).strip().lower().replace(" ", "_"): column
                    for column in df.columns
                }
                candidates = [
                    "review", "review_text", "text", "clean_text", "content", "comment"
                ]
                default_col = next(
                    (normalized[name] for name in candidates if name in normalized),
                    df.columns[0],
                )
                review_col = st.selectbox(
                    "Select review text column",
                    list(df.columns),
                    index=list(df.columns).index(default_col),
                )

                if st.button("🔎 Analyze all reviews", type="primary"):
                    texts = df[review_col].fillna("").astype(str)
                    valid_indices = [
                        index for index, text_value in enumerate(texts)
                        if text_value.strip()
                    ]

                    if not valid_indices:
                        st.warning("No non-empty reviews found in that column.")
                    else:
                        valid_texts = [
                            texts.iloc[index].strip() for index in valid_indices
                        ]
                        predictions = model.predict(valid_texts)

                        try:
                            scores = model.decision_function(valid_texts)
                        except Exception:
                            scores = [None] * len(valid_indices)

                        labels = ["Skipped: empty review"] * len(df)
                        score_values = [None] * len(df)

                        for index, prediction, score in zip(
                            valid_indices, predictions, scores
                        ):
                            labels[index] = (
                                "Potentially Deceptive"
                                if int(prediction) == 1
                                else "Likely Truthful"
                            )
                            score_values[index] = (
                                round(float(score), 4) if score is not None else None
                            )

                        results = df.copy()
                        results["Prediction"] = labels
                        results["SVM Decision Score"] = score_values
                        st.session_state["bulk_results"] = results

                if "bulk_results" in st.session_state:
                    results = st.session_state["bulk_results"]

                    st.subheader("Analysis Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Likely Truthful",
                        int((results["Prediction"] == "Likely Truthful").sum()),
                    )
                    col2.metric(
                        "Potentially Deceptive",
                        int((results["Prediction"] == "Potentially Deceptive").sum()),
                    )
                    col3.metric(
                        "Skipped empty rows",
                        int(results["Prediction"].astype(str).str.startswith("Skipped:").sum()),
                    )

                    st.dataframe(results, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download results CSV",
                        results.to_csv(index=False).encode("utf-8-sig"),
                        "fake_review_analysis_results.csv",
                        "text/csv",
                    )
                    st.caption(
                        "Decision scores are not probabilities. Predictions do not "
                        "prove that reviews are genuine or deceptive."
                    )

        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")


# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write(
        "Recorded results from the project's held-out test experiment. "
        "These values are not recalculated for new reviews."
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Accuracy (%)": 88.75,
                "Precision (%)": 86.90,
                "Recall (%)": 91.25,
                "F1-Score (%)": 89.02,
            },
            {
                "Model": "Multinomial Naive Bayes",
                "Accuracy (%)": 86.88,
                "Precision (%)": 83.91,
                "Recall (%)": 91.25,
                "F1-Score (%)": 87.43,
            },
            {
                "Model": "Linear SVM",
                "Accuracy (%)": 90.00,
                "Precision (%)": 89.02,
                "Recall (%)": 91.25,
                "F1-Score (%)": 90.12,
            },
        ]
    )

    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.success(
        "Linear SVM had the strongest recorded held-out accuracy and F1-score "
        "among these three evaluated models."
    )

    st.subheader("Linear SVM Confusion Matrix")
    confusion_matrix = pd.DataFrame(
        [[142, 18], [14, 146]],
        index=["Actual Truthful", "Actual Deceptive"],
        columns=["Predicted Truthful", "Predicted Deceptive"],
    )
    st.dataframe(confusion_matrix, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Correct Predictions", "288")
    col2.metric("Incorrect Predictions", "32")
    col3.metric("Test Samples", "320")

    st.subheader("Cross-validation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "86.99%")
    col2.metric("Precision", "87.20%")
    col3.metric("Recall", "86.88%")
    col4.metric("F1-Score", "86.98%")

    st.caption(
        "These are recorded evaluation results, not a guarantee of performance "
        "on every new review."
    )
