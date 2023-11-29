import json
import pandas as pd
import tensorflow as tf
import numpy as np
#import numpy as np
#import pandas as pd


#=========================================
# to parse a json
#=========================================
def load_quotes_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as jsonfile:
        quotes = json.load(jsonfile)
    return quotes

#=========================================
# to parse a txt file
#=========================================
def load_quotes_from_txt(txt_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        return lines

#=========================================
# to encode the characters 
#=========================================
def tokenize(text):
    # Create a mapping of unique characters to integer indices
    char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
    idx_to_char = np.array(list(char_to_idx.keys()))

    # Convert the text to a numerical representation
    text_as_int = np.array([char_to_idx[char] for char in text])

    return char_to_idx,idx_to_char,text_as_int

#=========================================
# to split the input data 
#=========================================
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#=========================================
# to generate a quote using the model 
#=========================================
def generate_text(model, start_string,char_to_idx, idx_to_char, num_generate=1000):

    input_eval = [char_to_idx[s] for s in start_string]

    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()

    for i in range(num_generate):

        predictions = model(input_eval)

        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx_to_char[predicted_id])

    return start_string + ''.join(text_generated)
           

#=========================================
# Setting up the model
#=========================================
def buildModel(char_to_idx, batch_size):
    vocab_size = len(char_to_idx)
    embedding_dim = 256
    rnn_units = 1024

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    return model

#=========================================
# Traning the model
#=========================================
def trainModel(model, dataset):

    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Number of training epochs
    EPOCHS = 20

    # Train the model
    model.fit(dataset, epochs=EPOCHS)   


# Example usage to load quotes
lines = load_quotes_from_txt('data/train.txt')

char_to_ind, ind_to_char, text_as_int = tokenize(lines)
seq_length = 100
examples_per_epoch = len(lines)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

batch_size=64

model = buildModel(char_to_idx=char_to_ind, batch_size=64)

trainModel(model,dataset)

# {
# # Display the loaded quotes
# for quote in loaded_quotes:
#     print(f"Quote: {quote['quote']}")
#     print(f"Author: {quote['author']}")
#     print(f"Categories: {', '.join(quote['categories'])}")
#     print()
# }



#{
#Create list of the quotes, ignoring authors and tags
# quotesSet= []
# for quote in loaded_quotes:
#     quotesSet.append(quote['quote'])

#Tokenize characters in quotes
# tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level = True)
# tokenizer.fit_on_texts(loaded_quotes)

#Checking to see how tokenizer works on first word
# sequences = tokenizer.texts_to_sequences([quotesSet])[0]
# print(sequences)
#}