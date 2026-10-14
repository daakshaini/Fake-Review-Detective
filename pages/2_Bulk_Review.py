import streamlit as st
import pandas as pd
import io
import hashlib
from common import apply_style, load_model

st.set_page_config(page_title="Bulk Review Analysis | Fake Review Detective", page_icon="📂", layout="wide")
apply_style()
model = load_model()

st.markdown('<div class="main-title">📂 Bulk Review Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Upload a CSV and analyze multiple hotel reviews in one batch.</div>', unsafe_allow_html=True)

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


st.markdown('<div class="footer">🔍 Fake Review Detective · Bulk Review Analysis</div>', unsafe_allow_html=True)
