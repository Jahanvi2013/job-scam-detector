
import streamlit as st
import pickle
import re


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🔍",
    layout="centered"
)


# ==========================================
# 2. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

@st.cache_resource
def load_model():

    with open("model/best_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("model/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================


def predict_job(text):

    cleaned_text = clean_text(text)

    transformed_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(transformed_text)[0]

    decision_score = model.decision_function(
        transformed_text
    )[0]

    # Get the model's vocabulary and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Calculate each feature's contribution
    feature_values = transformed_text.toarray()[0]

    contributions = feature_values * coefficients

    # Only retain features actually present in the posting
    present_indices = transformed_text.nonzero()[1]

    feature_results = []

    for index in present_indices:

        feature_results.append({
            "word": feature_names[index],
            "contribution": contributions[index]
        })

    # Sort features by their contribution
    fake_features = sorted(
        [f for f in feature_results if f["contribution"] > 0],
        key=lambda x: x["contribution"],
        reverse=True
    )

    legitimate_features = sorted(
        [f for f in feature_results if f["contribution"] < 0],
        key=lambda x: x["contribution"]
    )

    return (
        prediction,
        decision_score,
        fake_features[:5],
        legitimate_features[:5]
    )



# ==========================================
# 5. APPLICATION INTERFACE
# ==========================================

st.title("🔍 Job Scam Detector")

st.write(
    "Assess a job posting using a machine learning model "
    "trained to identify potentially fraudulent vacancies."
)

st.info(
    "This tool provides an initial screening assessment. "
    "It cannot definitively establish whether a job is genuine or fraudulent."
)


# ==========================================
# 6. JOB POSTING INPUT
# ==========================================

job_text = st.text_area(
    "Paste your job posting below",
    height=280,
    placeholder="Paste the job title, company profile, description, requirements and benefits..."
)


# ==========================================
# 7. ANALYSE BUTTON
# ==========================================

if st.button("Analyse Job Posting", type="primary"):

    if not job_text.strip():

        st.warning("Please enter a job posting first.")

    else:

        prediction, score, fake_features, legitimate_features = predict_job(job_text)

        st.divider()

        st.subheader("Assessment Result")

        if prediction == 1:

            st.error("⚠️ Potentially Fraudulent")

            st.write(
                "The model identified patterns more strongly "
                "associated with fraudulent postings in its training data."
            )

        else:

            st.success("✅ Likely Legitimate")

            st.write(
                "The model identified patterns more strongly "
                "associated with legitimate postings in its training data."
            )

        st.subheader("Model Decision Score")

        st.metric(
            label="Linear SVM Decision Score",
            value=f"{score:.3f}"
        )

        st.caption(
            "A score above zero indicates a fraudulent classification; "
            "a score below zero indicates a legitimate classification. "
            "This is a decision score, not a probability."
        )


        # ==========================================
        # MODEL EXPLAINABILITY
        # ==========================================

        st.divider()

        st.subheader("Why did the model make this prediction?")

        st.write(
            "The following words and phrases contributed "
            "to the model's decision for this particular posting."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Features pushing towards fraudulent**")

            if fake_features:

                for feature in fake_features:
                    st.write(
                        f"🔴 {feature['word']} "
                        f"({feature['contribution']:+.3f})"
                    )

            else:
                st.caption("No positive feature contributions found.")

        with col2:

            st.markdown("**Features pushing towards legitimate**")

            if legitimate_features:

                for feature in legitimate_features:
                    st.write(
                        f"🟢 {feature['word']} "
                        f"({feature['contribution']:+.3f})"
                    )

            else:
                st.caption("No negative feature contributions found.")

        st.caption(
            "These features explain the model's learned associations, "
            "not the actual cause or legitimacy of a job posting. "
            "They should not be interpreted as definitive fraud indicators."
        )



# ==========================================
# 8. ABOUT THE MODEL
# ==========================================

st.divider()

st.subheader("About this project")

st.write(
    """
    This application uses a Linear Support Vector Machine (SVM)
    with TF-IDF text features to classify job postings.

    Three machine learning algorithms were evaluated:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Linear Support Vector Machine

    Linear SVM was selected based on its overall performance.
    """
)

st.write(
    "**Test-set performance:**"
)

col1, col2 = st.columns(2)

col1.metric("Accuracy", "98.46%")
col2.metric("F1 Score", "83.87%")

col3, col4 = st.columns(2)

col3.metric("Precision", "85.12%")
col4.metric("Recall", "82.66%")
