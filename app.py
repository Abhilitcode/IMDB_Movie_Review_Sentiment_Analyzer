import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 


#load the tokenizer
tokenizer = joblib.load('tokenizer.pkl')

#load the model
@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model.keras')

model = load_lstm_model()

#give some title
st.title('IMDB Movie Review Sentiment Analyzer')
st.write('Enter a movie review to predict the sentiment')

#define the function
def predictive_system(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)  
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment
        
#get the user input
user_review = st.text_area('Enter your review about the movie')

#prediction button
if st.button('Predict Sentiment'):
    if user_review:
        sentiment = predictive_system(user_review)
        st.write(f"The predicted sentiment is: {sentiment}")
    else:
        st.write("Please enter the review to predict the sentiment!!")
    
    