## Import all the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load word index to convert the reviews into vectors
word_index=imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

## Load the model
model=load_model('simple_rnn_imdb.h5')

## function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2) + 3 for word in words]     
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review




## Create the streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a IMDB movie reviews to classify it as Positive or Negative')

## User review input
user_input=st.text_area('Enter your Movie review here')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    ##Make prediction
    prediction= model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    ## Dsiplay the prediction
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")

else:
    st.write('Please enter a review to classify')

