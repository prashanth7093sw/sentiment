
import streamlit as st
import joblib
# Load saved components
le = joblib.load("models/label_encoder.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
log_model = joblib.load("models/logistic_model.joblib")
nb_model = joblib.load("models/nb_model.joblib")

# Streamlit UI
st.title("Review Category Prediction App")
st.write("🔠 This app uses TF-IDF + ML to predict review categories (like Positive, Negative).")

# Text input
user_input = st.text_area("Enter cleaned review text here:")

# Model choice
model_option = st.selectbox("Choose the model to predict:", ["Logistic Regression", "Naive Bayes"])

# Predict
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        vector = tfidf.transform([user_input])

        # Predict
        if model_option == "Logistic Regression":
            pred = log_model.predict(vector)
        else:
            pred = nb_model.predict(vector)

        # Decode label
        label = le.inverse_transform(pred)[0]

        st.success(f"🧾 Predicted Review Category: *{label}*")
