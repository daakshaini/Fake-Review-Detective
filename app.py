import streamlit as st
import joblib
import pandas as pd
import io
import hashlib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fake Review Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    .stApp { background-color: #f5f7fb; }
    .block-container { max-width: 1150px; padding-top: 1.5rem; padding-bottom: 2rem; }
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

# ============================================================
# SIDEBAR
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
# HEADER / HERO
# ============================================================
st.markdown('<div class="main-title">🔍 Fake Review Detective</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">NLP and Machine Learning Based Hotel Review Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">Intelligent Hotel Review Analysis</div>'
    '<div class="hero-text">Analyze hotel review text using Natural Language Processing '
    'and Machine Learning to identify linguistic patterns associated with truthful and '
    'potentially deceptive reviews.</div>'
    '<div class="hero-tag">NLP &nbsp;•&nbsp; TF-IDF &nbsp;•&nbsp; Linear SVM</div>'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Dataset", "1,596", "Unique reviews")
with col2:
    st.metric("Model", "Linear SVM", "Final classifier")
with col3:
    st.metric("Accuracy", "90.00%", "Held-out test set")
with col4:
    st.metric("F1-Score", "90.12%", "Held-out test set")

st.divider()
st.markdown('<div class="section-title">📝 Review Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-text">Select a hotel, enter its review, and analyze the review text.</div>', unsafe_allow_html=True)

# ============================================================
# REVIEW INPUT
# ============================================================
with st.container(border=True):
    st.markdown("### 🏨 Select Hotel")
    hotels = [
        "Affinia", "Allegro", "Amalfi", "Ambassador", "Conrad", "Fairmont",
        "Hard Rock", "Hilton", "Homewood Suites", "Hyatt", "InterContinental",
        "James", "Knickerbocker", "Monaco", "Omni", "Palmer House", "Sheraton",
        "Sofitel", "Swissotel", "Talbott",
    ]
    selected_hotel = st.selectbox("Choose the hotel being reviewed:", hotels)
    st.markdown("### ✍️ Enter Hotel Review")
    sample_reviews = {
        "No sample selected": "",
        "Sample 1 — Positive Experience": (
            "The hotel was comfortable and clean. The room was spacious, the staff were polite, "
            "and the location was convenient. I enjoyed my stay and would consider returning."
        ),
        "Sample 2 — Highly Exaggerated": (
            "This hotel is absolutely perfect in every possible way! Everything was unbelievably "
            "amazing and the staff were the best I have ever seen. I cannot say enough wonderful "
            "things about this incredible hotel and everyone should stay here immediately!"
        ),
        "Sample 3 — Detailed Experience": (
            "I stayed at this hotel for three nights. The check-in process was quick, the room was "
            "clean, and the bed was comfortable. The staff answered my questions and the hotel was "
            "close to several restaurants."
        ),
    }
    selected_sample = st.selectbox("🧪 Try a sample review:", list(sample_reviews.keys()))
    review = st.text_area(
        "Review",
        value=sample_reviews[selected_sample],
        placeholder="Paste or type your hotel review here...",
        height=210,
        label_visibility="collapsed",
    )
    analyze_button = st.button("🔎 Analyze Review", use_container_width=True)

# ============================================================
# PREDICTION + EXPLAINABLE AI
# ============================================================
if analyze_button:
    if not review.strip():
        st.warning("⚠️ Please enter a hotel review before analyzing.")
    else:
        prediction = model.predict([review])[0]
        decision_score = float(model.decision_function([review])[0])

        st.markdown('<div class="section-title">📊 Analysis Result</div>', unsafe_allow_html=True)
        st.write(f"**Hotel selected:** {selected_hotel}")

        if prediction == 1:
            st.markdown(
                '<div class="deceptive-result"><div class="deceptive-title">⚠️ Potentially Deceptive</div>'
                '<div class="result-text">The model detected linguistic patterns associated with deceptive reviews in its training data.</div></div>',
                unsafe_allow_html=True,
            )
            direction = "Deceptive"
        else:
            st.markdown(
                '<div class="truth-result"><div class="truth-title">✅ Likely Truthful</div>'
                '<div class="result-text">The model detected linguistic patterns associated with truthful reviews in its training data.</div></div>',
                unsafe_allow_html=True,
            )
            direction = "Truthful"

        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric("SVM Decision Score", f"{decision_score:.4f}")
        with score_col2:
            st.metric("Decision Direction", direction)

        st.info(
            "The selected hotel is contextual information. The classifier predicts from review text only. "
            "The SVM score is not a probability or confidence percentage."
        )

        with st.expander("🔎 Why did the model make this prediction?"):
            try:
                tfidf_step = model.named_steps["tfidf"]
                svm_step = model.named_steps["model"]
                review_vector = tfidf_step.transform([review])
                feature_names = tfidf_step.get_feature_names_out()
                coefficients = svm_step.coef_[0]
                contributions = review_vector.multiply(coefficients).toarray()[0]
                present_indices = review_vector.indices

                toward_deceptive = sorted(
                    [(feature_names[i], float(contributions[i])) for i in present_indices if contributions[i] > 1e-8],
                    key=lambda item: item[1], reverse=True,
                )[:5]
                toward_truthful = sorted(
                    [(feature_names[i], float(contributions[i])) for i in present_indices if contributions[i] < -1e-8],
                    key=lambda item: item[1],
                )[:5]

                if prediction == 1:
                    supporting_features, opposing_features = toward_deceptive, toward_truthful
                else:
                    supporting_features, opposing_features = toward_truthful, toward_deceptive

                explain_left, explain_right = st.columns(2)
                with explain_left:
                    st.markdown("**Patterns supporting the prediction**")
                    if supporting_features:
                        for term, value in supporting_features:
                            st.write(f"• `{term}` — contribution: `{abs(value):.4f}`")
                    else:
                        st.write("No strong individual word contribution was found in this review.")
                with explain_right:
                    st.markdown("**Patterns pushing the other way**")
                    if opposing_features:
                        for term, value in opposing_features:
                            st.write(f"• `{term}` — contribution: `{abs(value):.4f}`")
                    else:
                        st.write("No strong opposing word contribution was found in this review.")
                st.caption(
                    "Contributions show how the review's TF-IDF features affect the linear SVM decision score. "
                    "They are not proof of truthfulness or deception."
                )
            except (AttributeError, KeyError, ValueError):
                st.warning("The prediction was generated, but the feature explanation could not be calculated for this model.")

# ============================================================
# BULK REVIEW ANALYSIS
# ============================================================
st.divider()
st.markdown('<div class="section-title">📂 Bulk Review Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-text">Upload a CSV with multiple reviews, classify them in one batch, and download the results.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("### Upload multiple hotel reviews")
    st.write(
        "Your CSV must contain a column with review text. A hotel column is optional; "
        "hotel names are included as context only and are not used by the classifier."
    )

    template_df = pd.DataFrame(
        {
            "hotel": ["Hilton", "Conrad", "Hyatt"],
            "review": [
                "The room was clean and check-in was quick. The staff were helpful.",
                "The bathroom was dirty and the air conditioner did not work properly.",
                "I stayed for two nights and the room was comfortable, although the Wi-Fi was slow.",
            ],
        }
    )
    st.download_button(
        "⬇️ Download sample CSV template",
        data=template_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="fake_review_detective_template.csv",
        mime="text/csv",
        key="bulk_template_download",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Use one row per review. Include a column such as review, review_text, or text.",
        key="bulk_review_csv_upload",
    )

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        upload_digest = hashlib.sha256(upload_bytes).hexdigest()

        try:
            bulk_df = pd.read_csv(io.BytesIO(upload_bytes))
        except Exception as exc:
            bulk_df = None
            st.error(f"Could not read this CSV file. Please check its format. Details: {exc}")

        if bulk_df is not None:
            if bulk_df.empty:
                st.warning("The uploaded CSV has no data rows.")
            elif len(bulk_df) > 5000:
                st.warning("For this demo, please upload a CSV with no more than 5,000 rows at a time.")
            elif len(bulk_df.columns) == 0:
                st.error("The uploaded CSV does not contain any columns.")
            else:
                st.caption(f"Loaded {len(bulk_df):,} rows and {len(bulk_df.columns)} columns.")
                st.markdown("**Preview**")
                st.dataframe(bulk_df.head(5), use_container_width=True, hide_index=True)

                normalized_columns = {
                    str(col).strip().lower().replace(" ", "_"): col
                    for col in bulk_df.columns
                }
                review_candidates = ["review", "review_text", "text", "clean_text", "content", "comment"]
                detected_review_col = next(
                    (normalized_columns[name] for name in review_candidates if name in normalized_columns),
                    bulk_df.columns[0],
                )
                review_col_index = list(bulk_df.columns).index(detected_review_col)
                review_col = st.selectbox(
                    "Select the column containing review text:",
                    options=list(bulk_df.columns),
                    index=review_col_index,
                    key="bulk_review_text_column",
                )

                hotel_candidates = ["hotel", "hotel_name", "property", "property_name"]
                detected_hotel_col = next(
                    (normalized_columns[name] for name in hotel_candidates if name in normalized_columns and normalized_columns[name] != review_col),
                    None,
                )
                hotel_options = ["(No hotel column)"] + [col for col in bulk_df.columns if col != review_col]
                default_hotel_index = 0
                if detected_hotel_col in hotel_options:
                    default_hotel_index = hotel_options.index(detected_hotel_col)
                hotel_col = st.selectbox(
                    "Hotel column (optional):",
                    options=hotel_options,
                    index=default_hotel_index,
                    key="bulk_hotel_context_column",
                )

                if st.button("🔎 Analyze all reviews", use_container_width=True, key="run_bulk_review_analysis"):
                    review_series = bulk_df[review_col].fillna("").astype(str)
                    valid_positions = [
                        position
                        for position, value in enumerate(review_series.tolist())
                        if value.strip()
                    ]

                    if not valid_positions:
                        st.warning("No non-empty reviews were found in the selected column.")
                    else:
                        valid_texts = [review_series.iloc[position].strip() for position in valid_positions]
                        try:
                            batch_predictions = model.predict(valid_texts)
                            batch_scores = model.decision_function(valid_texts)

                            labels = ["Skipped: empty review"] * len(bulk_df)
                            scores = [None] * len(bulk_df)

                            for position, predicted_label, score in zip(
                                valid_positions, batch_predictions, batch_scores
                            ):
                                labels[position] = (
                                    "Potentially Deceptive" if predicted_label == 1 else "Likely Truthful"
                                )
                                scores[position] = round(float(score), 4)

                            results_df = bulk_df.copy()
                            results_df["Prediction"] = labels
                            results_df["SVM Decision Score"] = scores

                            st.session_state["bulk_analysis_result_df"] = results_df
                            st.session_state["bulk_analysis_result_hash"] = upload_digest
                            st.session_state["bulk_analysis_review_col"] = review_col
                            st.session_state["bulk_analysis_hotel_col"] = hotel_col
                        except Exception as exc:
                            st.error(f"Batch analysis failed. Details: {exc}")

                # Keep the results visible during Streamlit reruns (for example, when downloading).
                if (
                    st.session_state.get("bulk_analysis_result_hash") == upload_digest
                    and "bulk_analysis_result_df" in st.session_state
                ):
                    results_df = st.session_state["bulk_analysis_result_df"]
                    truthful_count = int((results_df["Prediction"] == "Likely Truthful").sum())
                    deceptive_count = int((results_df["Prediction"] == "Potentially Deceptive").sum())
                    skipped_count = int(results_df["Prediction"].astype(str).str.startswith("Skipped:").sum())
                    analyzed_count = truthful_count + deceptive_count

                    st.markdown("### 📊 Bulk Analysis Summary")
                    bulk_metric1, bulk_metric2, bulk_metric3, bulk_metric4 = st.columns(4)
                    with bulk_metric1:
                        st.metric("Total rows", f"{len(results_df):,}")
                    with bulk_metric2:
                        st.metric("Likely Truthful", f"{truthful_count:,}")
                    with bulk_metric3:
                        st.metric("Potentially Deceptive", f"{deceptive_count:,}")
                    with bulk_metric4:
                        st.metric("Skipped empty rows", f"{skipped_count:,}")

                    if analyzed_count:
                        deceptive_share = int(round(100 * deceptive_count / analyzed_count))
                        st.caption(f"Among {analyzed_count:,} analyzed reviews, {deceptive_share}% were classified as potentially deceptive by the model. This is a prediction share, not measured accuracy.")
                        st.progress(deceptive_share, text="Share predicted potentially deceptive")

                    st.markdown("### Detailed results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download analysis results (CSV)",
                        data=results_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="fake_review_analysis_results.csv",
                        mime="text/csv",
                        key="bulk_results_download",
                    )
                    st.caption(
                        "The classifier predicts from review text only. Decision scores are not probabilities. "
                        "A classification does not prove that a review is truthful or deceptive."
                    )

# ============================================================
# WORKFLOW
# ============================================================
st.divider()
st.markdown('<div class="section-title">⚙️ How the System Works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-text">The review follows the machine-learning pipeline below.</div>', unsafe_allow_html=True)
w1, w2, w3, w4, w5 = st.columns(5)
workflow_data = [
    ("🏨", "Hotel", "Select hotel"),
    ("📝", "Review", "Enter review"),
    ("🔤", "TF-IDF", "Text to features"),
    ("🤖", "Linear SVM", "Classification"),
    ("📊", "Prediction", "Result displayed"),
]
for column, data in zip([w1, w2, w3, w4, w5], workflow_data):
    icon, name, description = data
    with column:
        st.markdown(
            '<div class="workflow">'
            f'<div class="workflow-icon">{icon}</div>'
            f'<div class="workflow-name">{name}</div>'
            f'<div class="workflow-text">{description}</div></div>',
            unsafe_allow_html=True,
        )

# ============================================================
# MODEL PERFORMANCE DASHBOARD (NO st.bar_chart)
# Built-in progress bars avoid the local Narwhals/Altair error.
# ============================================================
st.divider()
st.markdown('<div class="section-title">📈 Model Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-text">Explore recorded held-out evaluation results, confusion matrix, and cross-validation metrics.</div>',
    unsafe_allow_html=True,
)

with st.expander("📊 Open Model Performance Dashboard", expanded=False):
    comparison_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy (%)": 88.75, "Precision (%)": 86.90, "Recall (%)": 91.25, "F1-Score (%)": 89.02},
        {"Model": "Multinomial Naive Bayes", "Accuracy (%)": 86.88, "Precision (%)": 83.91, "Recall (%)": 91.25, "F1-Score (%)": 87.43},
        {"Model": "Linear SVM", "Accuracy (%)": 90.00, "Precision (%)": 89.02, "Recall (%)": 91.25, "F1-Score (%)": 90.12},
    ])
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Confusion Matrix", "Cross-Validation"])

    with tab1:
        st.markdown("### Model Evaluation Results")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.markdown("### Accuracy and F1-Score Comparison")
        for _, row in comparison_df.iterrows():
            st.markdown(f"**{row['Model']}**")
            left, right = st.columns(2)
            with left:
                st.caption(f"Accuracy: {row['Accuracy (%)']:.2f}%")
                st.progress(min(100, max(0, int(round(row['Accuracy (%)'])))))
            with right:
                st.caption(f"F1-Score: {row['F1-Score (%)']:.2f}%")
                st.progress(min(100, max(0, int(round(row['F1-Score (%)'])))))
        st.success("Linear SVM achieved the strongest recorded held-out test accuracy and F1-score among these three evaluated models.")
        st.caption("Recorded results from the original test experiment; these are not recalculated for new user reviews.")

    with tab2:
        st.markdown("### Linear SVM Confusion Matrix")
        confusion_df = pd.DataFrame(
            [[142, 18], [14, 146]],
            index=["Actual Truthful", "Actual Deceptive"],
            columns=["Predicted Truthful", "Predicted Deceptive"],
        )
        st.dataframe(confusion_df, use_container_width=True)
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            st.metric("Correct Predictions", "288")
        with cm2:
            st.metric("Incorrect Predictions", "32")
        with cm3:
            st.metric("Test Samples", "320")
        st.info("Rows are actual labels; columns are predicted labels. This is the recorded held-out evaluation matrix.")

    with tab3:
        st.markdown("### Five-Fold Cross-Validation")
        cv1, cv2, cv3, cv4 = st.columns(4)
        with cv1:
            st.metric("Accuracy", "86.99%")
        with cv2:
            st.metric("Precision", "87.20%")
        with cv3:
            st.metric("Recall", "86.88%")
        with cv4:
            st.metric("F1-Score", "86.98%")
        st.markdown("### Recorded Fold Accuracies")
        fold_df = pd.DataFrame({"Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"], "Accuracy (%)": [91.02, 84.31, 85.88, 85.88, 87.84]})
        st.dataframe(fold_df, use_container_width=True, hide_index=True)
        for _, row in fold_df.iterrows():
            st.caption(f"{row['Fold']}: {row['Accuracy (%)']:.2f}%")
            st.progress(min(100, max(0, int(round(row['Accuracy (%)'])))))
        st.caption("Cross-validation results are separate from the held-out test-set metrics.")

# ============================================================
# PROJECT DETAILS
# ============================================================
st.divider()
st.markdown('<div class="section-title">📚 Project Details</div>', unsafe_allow_html=True)
detail1, detail2 = st.columns(2)
with detail1:
    with st.expander("🤖 Model Evaluation"):
        st.write(
            """Algorithms evaluated:

            • Logistic Regression
            • Multinomial Naive Bayes
            • Linear SVM

            TF-IDF converts review text into numerical features. Linear SVM was selected based on its recorded held-out test performance."""
        )
with detail2:
    with st.expander("⚠️ Limitations"):
        st.write(
            """• The model was trained specifically on hotel reviews.

            • The system primarily learns patterns from review text.

            • It cannot verify whether the reviewer stayed at the hotel.

            • Negative sentiment does not automatically imply deception.

            • A truthful review may be misclassified.

            • The hotel dropdown is contextual information, not an ML prediction feature.

            • A prediction does not prove that a review is fake or genuine."""
        )

st.divider()
st.warning(
    "Important: This application provides a machine-learning prediction based on linguistic patterns learned from labeled data. "
    "It does not independently establish whether a review is truthful or deceptive."
)
st.markdown(
    '<div class="footer">🔍 Fake Review Detective<br>NLP + TF-IDF + Linear SVM + Explainable AI<br>Academic Data Science Project</div>',
    unsafe_allow_html=True,
)
