from flask import Flask, jsonify, request 
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
model = pickle.load(open("model.pkl",'rb'))
tokenizer = pickle.load(open("token.pkl",'rb'))
app = Flask(__name__)
CORS(app) 
@app.route('/') 
def index():
    return "<center><h1>To use the API put / followed by your text on the root url!</h1></center>"

max_len=20

@app.route('/<st>', methods = ['GET']) 
def detect(st):
    input_text = request.args.get('in')
    rs = predict_sentiment(st)
    return jsonify({
                    'prediction' : rs,
                    }) 
  




def preprocess_input_text(text):
    # Function to preprocess the input text
    stop = stopwords.words('english')
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'#[a-zA-Z]+|\$[a-zA-Z]+|@[a-zA-Z]+|[,.^_$*%-;鶯!?:]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop])
    return text


def predict_sentiment(text):
    preprocessed_text = preprocess_input_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=max_len)
    prediction = model.predict(padded_sequence)
    sentiment_label = np.argmax(prediction)
    if sentiment_label == 0:
        return "Negative"
    elif sentiment_label == 1:
        return "Neutral"
    elif sentiment_label == 2:
        return "Positive"
input_text = "The movie was very good"
predicted_sentiment = predict_sentiment(input_text)
print("Predicted Sentiment:", predicted_sentiment)



if __name__ == "__main__":   
    app.run(debug=False)   
       