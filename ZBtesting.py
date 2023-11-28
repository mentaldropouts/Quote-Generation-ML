import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load and preprocess your data
with open('data/quotes.json', 'r', encoding='utf-8') as jsonfile:
    quotes_data = json.load(jsonfile)

quotes_text = [quote['quote'] for quote in quotes_data]

# Tokenize the data with a limited vocabulary size
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(quotes_text)

total_words = min(500, len(tokenizer.word_index) + 1)

# Create input sequences and labels
input_sequences = []
for line in quotes_text:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Build an RNN model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, verbose=1)

# Save the trained model
model.save("quote_generation_rnn_model")
