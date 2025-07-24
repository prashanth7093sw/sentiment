import streamlit as st
import joblib
import pandas as pd
import os

st.title("🧠 NLP Sentiment Classifier")
st.markdown("Enter a review below and get the predicted sentiment using saved models.")

# Load models (from current directory)
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer_cleaned_text.joblib")
    label_encoder = joblib.load("label_encoder_review_category.joblib")
    model = joblib.load("lr_model.joblib")
    model = joblib.load("nb_model.joblib")  # or switch to 'nb_model.joblib'
    return tfidf, label_encoder, model

tfidf, label_encoder, model = load_models()

# Input
review = st.text_area("✍️ Enter your review text:")

if st.button("🔍 Predict"):
    if not review.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        vector = tfidf.transform([review])
        prediction = model.predict(vector)
        sentiment = label_encoder.inverse_transform(prediction)[0]
        
        st.success(f"🎯 Sentiment: **{sentiment}**")
        
        # Optional download
        df_result = pd.DataFrame({
            "Input": [review],
            "Predicted Sentiment": [sentiment]
        })
        st.download_button(
            label="📥 Download Result",
            data=df_result.to_csv(index=False),
            file_name="sentiment_prediction.csv",
            mime="text/csv"
        )

