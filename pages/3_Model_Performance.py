import streamlit as st
import pandas as pd
from common import apply_style

st.set_page_config(page_title="Model Performance | Fake Review Detective", page_icon="📈", layout="wide")
apply_style()

st.markdown('<div class="main-title">📈 Model Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Recorded evaluation results from the project experiments.</div>', unsafe_allow_html=True)

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


st.caption("These are recorded evaluation results from the project experiment; they are not recalculated for new user reviews.")
