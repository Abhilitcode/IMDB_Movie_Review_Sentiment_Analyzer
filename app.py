import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn
import threading
import requests

# Initialize FastAPI app
app = FastAPI()

# Load the tokenizer and model
tokenizer = joblib.load('tokenizer.pkl')

#load the model
@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model.keras')

model = load_model('lstm_model.keras')

# Pydantic model for request validation
class Review(BaseModel):
    text: str

# Prediction function
def predictive_system(review_text):
    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment

# Define the FastAPI prediction endpoint
@app.post("/predict")

#async is required when you get data from 2 url. here only def will work. 
async def predict_sentiment(data: Review):
    try:
        sentiment = predictive_system(data.text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start FastAPI in a background thread
def start_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

threading.Thread(target=start_fastapi, daemon=True).start()

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review to predict the sentiment")

user_review = st.text_area("Enter your review about the movie")

if st.button("Predict Sentiment"):
    if user_review:
        # Send review text to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_review})
        
        if response.status_code == 200:
            sentiment = response.json().get("sentiment")
            st.write(f"The predicted sentiment is: {sentiment}")
        else:
            st.write("Error in prediction. Please try again.")
    else:
        st.write("Please enter a review to predict the sentiment!")

    
    