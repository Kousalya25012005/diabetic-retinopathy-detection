
            col2.metric(
                "Model Score",
                f"{result['Score']:.2f}%"
            )

            st.success(result["Diagnosis"])
            st.write(GRADE_INFO[result["Grade"]])

            st.caption(
                "The model score is not a calibrated probability "
                "or a measure of diagnostic certainty."
            )

            st.subheader("Scores Across All Five Grades")

            score_df = pd.DataFrame({
                "Grade": [f"Grade {i}" for i in range(5)],
                "Model Score (%)": result["Grade Scores"]
            })

            st.bar_chart(score_df.set_index("Grade"))

            report = f"""
DIABETIC RETINOPATHY SCREENING REPORT

Image: {result['Image']}
Date: {result['Time']}

Predicted Grade: {result['Grade']}
Prediction: {result['Diagnosis']}
Model Score: {result['Score']}%

IMPORTANT:
This is an AI research prototype, not a medical diagnosis.
Consult a qualified ophthalmologist for clinical assessment.
"""

            st.download_button(
                "📥 Download Prediction Report",
                data=report,
                file_name="DR_prediction_report.txt",
                mime="text/plain"
            )

    st.warning(
        "This application is for research and educational use only. "
        "It cannot diagnose or rule out diabetic retinopathy."
    )

# ---------------- HISTORY & ANALYTICS ----------------

elif page == "History & Analytics":
    st.title("📊 Prediction History & Analytics")

    history = st.session_state.history

    if len(history) == 0:
        st.info(
            "No prediction history available. Analyze an image first."
        )
    else:
        df = pd.DataFrame(history)

        st.subheader("Prediction Summary")

        col1, col2 = st.columns(2)

        col1.metric("Total Predictions", len(df))

        col2.metric(
            "Average Model Score",
            f"{df['Score'].mean():.2f}%"
        )

        st.divider()
        st.subheader("Grade Distribution")

        grade_counts = (
            df["Grade"]
            .value_counts()
            .reindex(range(5), fill_value=0)
        )

        chart_df = pd.DataFrame({
            "Grade": [f"Grade {i}" for i in range(5)],
            "Number of Predictions": grade_counts.values
        })

        st.bar_chart(chart_df.set_index("Grade"))

        st.divider()
        st.subheader("Previous Predictions")

        display_df = df.drop(columns=["Grade Scores"], errors="ignore")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download History (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="DR_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            if "last_prediction" in st.session_state:
                del st.session_state.last_prediction
            st.rerun()

# ---------------- FOOTER ----------------

st.divider()

st.caption("RetinaAI | EfficientNetB0 | APTOS 2019 Dataset")
st.caption("Research prototype only. Not intended for clinical diagnosis.")
