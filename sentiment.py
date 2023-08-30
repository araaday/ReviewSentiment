from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# Load the model, tokenizer, and label binarizer
loaded_model = tf.keras.models.load_model('model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_binarizer = pickle.load(open('label.pkl', 'rb'))

# Define your cleaning function here
def cleaning(text):
    # Remove punctuations and convert to lowercase
    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # Remove stopwords
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]

    # Lemmatize the words
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    review_text = data['review']

    # Preprocess the review text
    cleaned_review = cleaning(review_text)
    review_seq = tokenizer.texts_to_sequences([cleaned_review])
    review_padded = pad_sequences(review_seq)

    # Make predictions
    predictions = loaded_model.predict(review_padded)
    predicted_label = label_binarizer.classes_[predictions.argmax(axis=1)]
    return jsonify({'predicted_rating': predicted_label[0]})


if __name__ == '__main__':
    app.run(debug=True)