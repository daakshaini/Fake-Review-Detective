import streamlit as st
from common import apply_style, load_model

st.set_page_config(page_title="Single Review Analysis | Fake Review Detective", page_icon="📝", layout="wide")
apply_style()
model = load_model()

st.markdown('<div class="main-title">📝 Single Review Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Analyze one hotel review and inspect the model explanation.</div>', unsafe_allow_html=True)

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


st.markdown('<div class="footer">🔍 Fake Review Detective · Single Review Analysis</div>', unsafe_allow_html=True)
