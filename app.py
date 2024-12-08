# import streamlit as st
# import re
# import pickle
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import nltk
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Define the LSTM model class (same as your training code)
# class SentimentLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(SentimentLSTM, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         x = self.embedding(x)
#         x, _ = self.lstm(x)
#         x = self.dropout(x[:, -1, :])
#         x = self.fc(x)
#         return x

# # Load pre-trained model and other necessary components
# def load_model():
#     model = SentimentLSTM(vocab_size=5000, embedding_dim=128, hidden_dim=256, output_dim=3)  # Adjust parameters as needed
#     model.load_state_dict(torch.load('sentiment_lstm_model.pth', map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # Load tokenizer and label encoder
# # tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
# # label_encoder = LabelEncoder()

# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# with open('label_encoder.pickle', 'rb') as handle:
#     label_encoder = pickle.load(handle)

# # Load NLTK stopwords
# nltk.download('stopwords')
# stop_words = set(nltk.corpus.stopwords.words('english'))

# # Function to preprocess the tweet text
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'#\w+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = " ".join([word for word in text.split() if word not in stop_words])
#     return text

# # Function to predict sentiment of the tweet
# def predict_sentiment(tweet, model):
#     # Preprocess the text
#     tweet = preprocess_text(tweet)
    
#     # Convert text to sequence
#     seq = tokenizer.texts_to_sequences([tweet])
#     padded = pad_sequences(seq, maxlen=50, padding='post')
    
#     # Convert to tensor
#     input_tensor = torch.tensor(padded, dtype=torch.long)
    
#     # Make prediction
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted = torch.max(output, 1)
        
#     # Decode the prediction
#     sentiment = label_encoder.inverse_transform(predicted.cpu().numpy())
#     return sentiment[0]

# # Streamlit app UI
# st.title("Sentiment Analysis on Tweets")
# st.write("Enter a tweet to predict its sentiment.")

# # User input for the tweet
# tweet_input = st.text_area("Tweet:", "")

# if st.button('Predict Sentiment') and tweet_input:
#     model = load_model()  # Load the trained model
#     sentiment = predict_sentiment(tweet_input, model)
#     st.write(f"Predicted Sentiment: {sentiment}")
import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re

# Load pre-trained model and necessary components
def load_model():
    # Load the model
    model = SentimentLSTM(vocab_size=5000, embedding_dim=128, hidden_dim=256, output_dim=3)  # Adjust output_dim based on your sentiment classes
    model.load_state_dict(torch.load('sentiment_lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load tokenizer and label encoder
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Define the LSTM model class (same as in training script)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Preprocess text (remove URLs, mentions, special characters)
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Function to predict sentiment of the tweet
def predict_sentiment(tweet, model):
    tweet = preprocess_text(tweet)  # Preprocess the tweet
    seq = tokenizer.texts_to_sequences([tweet])  # Convert text to sequence
    padded = pad_sequences(seq, maxlen=50, padding='post')  # Pad sequence
    
    # Convert to tensor
    input_tensor = torch.tensor(padded, dtype=torch.long)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    # Decode the prediction
    sentiment = label_encoder.inverse_transform(predicted.cpu().numpy())
    return sentiment[0]

# Streamlit app UI
st.title("Sentiment Analysis on Tweets")
st.write("Enter a tweet to predict its sentiment.")

# User input for the tweet
tweet_input = st.text_area("Tweet:", "")

if st.button('Predict Sentiment') and tweet_input:
    model = load_model()  # Load the trained model
    sentiment = predict_sentiment(tweet_input, model)
    st.write(f"Predicted Sentiment: {sentiment}")
