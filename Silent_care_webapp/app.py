import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from textblob import TextBlob
import os
import pickle
import re
import random
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load data from intent.json
json_file_path = 'intent.json'
def load_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)['intents']
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}.")
        return []

json_intents = load_json_data(json_file_path)
def intents_to_df(intents, source="json"):
    dic = {"tag": [], "patterns": [], "responses": []}
    for intent in intents:
        tag = f"{source}_{intent['tag']}"
        for pattern in intent['patterns']:
            if pattern.strip():
                dic['tag'].append(tag)
                dic['patterns'].append(pattern)
                dic['responses'].append(intent['responses'])
    return pd.DataFrame.from_dict(dic)

# Combine data (only from intent.json)
combined_df = intents_to_df(json_intents)
if combined_df.empty:
    print("Warning: combined_df is empty. Check intent.json.")

# Load or train model
model_path = "chatbot_model.h5"
tokenizer_path = "tokenizer.pkl"
lbl_enc_path = "labelencoder.pkl"

model_files_exist = os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(lbl_enc_path)

if model_files_exist:
    print("Loading saved model...")
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(lbl_enc_path, 'rb') as f:
        lbl_enc = pickle.load(f)
else:
    print("Training new model...")
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(combined_df['patterns'])
    vocab_size = len(tokenizer.word_index) + 1
    ptrn2seq = tokenizer.texts_to_sequences(combined_df['patterns'])
    X = pad_sequences(ptrn2seq, padding='post')
    lbl_enc = LabelEncoder()
    y = lbl_enc.fit_transform(combined_df['tag'])
    non_empty_idx = np.sum(X, axis=1) != 0
    X, y = X[non_empty_idx], y[non_empty_idx]
    X, y = X.astype('int32'), y.astype('int32')

    model = Sequential([
        Input(shape=(X.shape[1],)),
        Embedding(input_dim=vocab_size, output_dim=100, mask_zero=True),
        LSTM(64),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(len(np.unique(y)), activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, batch_size=10, validation_data=(X_val, y_val), epochs=30, verbose=1)

    model.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(lbl_enc_path, 'wb') as f:
        pickle.dump(lbl_enc, f)
    print(f"Model saved to {model_path}")

tokenizer.fit_on_texts(combined_df['patterns'])

# Load feedback (simplified for web)
feedback_data = {"feedback": [], "overrides": {}}

# Chatbot logic
def extract_keywords(text):
    words = re.sub('[^a-zA-Z\']', ' ', text.lower()).split()
    stop_words = {'i', 'am', 'the', 'a', 'an', 'and', 'or', 'to', 'in', 'is', 'are', 'you'}
    return [word for word in words if word not in stop_words and len(word) > 2][:3]

def generate_answer(pattern):
    text = [re.sub('[^a-zA-Z\']', ' ', pattern.lower()).strip()]
    x_test = tokenizer.texts_to_sequences(text)
    if not x_test[0]:
        return "Sorry, I didn’t understand you.", 0.00

    x_test = pad_sequences(x_test, padding='post', maxlen=model.input_shape[1])
    y_pred = model.predict(x_test, verbose=0)
    confidence = np.max(y_pred) * 100
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    print(f"Predicted tag: {tag}")

    matching_rows = combined_df[combined_df['tag'] == tag]
    sentiment = TextBlob(pattern).sentiment.polarity

    # Special handling for critical intents
    if any(phrase in pattern.lower() for phrase in ["kill myself", "want to die", "commit suicide"]):
        return "I’m very sorry to hear that but you have so much to look forward to. Please seek help by contacting: 9152987821.", 95.0

    if matching_rows.empty:
        print(f"Error: Tag '{tag}' not found in combined_df.")
        if sentiment < -0.2:
            return "I’m really sorry you’re feeling this way. I’m here to listen—want to tell me more?", confidence
        elif sentiment > 0.2:
            return "That’s great to hear! How can I assist you today?", confidence
        else:
            return "I’m not sure what you mean, but I’m here to help. What’s on your mind?", confidence

    responses = matching_rows['responses'].values[0]
    keywords = extract_keywords(pattern)
    pattern_key = pattern.lower()

    if pattern_key in feedback_data["overrides"]:
        response = feedback_data["overrides"][pattern_key]["response"]
        confidence = feedback_data["overrides"][pattern_key]["confidence"]
    else:
        available_responses = [r for r in responses if pattern_key not in feedback_data["overrides"] or r not in feedback_data["overrides"][pattern_key].get("excluded", [])]
        base_response = random.choice(available_responses) if available_responses else "I’m here to help. What’s on your mind?"
        
        if sentiment < -0.2:
            prefix = "I’m so sorry to hear that."
        elif sentiment > 0.2:
            prefix = "That’s wonderful!"
        else:
            prefix = "Got it."
        response = f"{prefix} {base_response}"

    return response, confidence

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response, confidence = generate_answer(user_input)
    return jsonify({'response': response, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)