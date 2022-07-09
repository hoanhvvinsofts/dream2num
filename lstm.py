from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from underthesea import word_tokenize

import tensorflow as tf
import numpy as np
import pandas as pd

def plot_graphs(history, string):
    import matplotlib.pyplot as plt
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

def test():
    # Processing data
    col_list = ["stt", "word", "lucky_number"]
    dreamwords = pd.read_csv("dreambook_preprocessed.csv", usecols=col_list)
    dreamwords_list = list(dreamwords["word"])
    number_list = list(dreamwords["lucky_number"])
    
    corpus = []
    for sentence in dreamwords_list:
        if type(sentence) is float:
            sentence = ""
            
        corpus.append(word_tokenize(sentence, format="text"))
    
    # Fit token
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    # for line in corpus:
    for i, line in enumerate(corpus):
        token_list = tokenizer.texts_to_sequences([line])[0] # [460, 461]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            
        input_sequences.append(n_gram_sequence)
    
    # Train
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    xs = input_sequences
    
    # Labels
    unique_labels = set()
    total_labels = [[unique_labels.add(number) for number in numbers.split("-")] for numbers in number_list]
    len_label = len(unique_labels)
    
    max_label_len = max([len(numbers.split("-")) for numbers in number_list])
    
    labels = [[number for number in numbers.split("-")] for numbers in number_list]
    labels = np.array(pad_sequences(labels, maxlen=max_label_len, padding='pre'))

    ys = labels
    # ys = tf.keras.utils.to_categorical(labels, num_classes=len_label)
    
    # Gerenate Model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(max_label_len, activation='softmax'))
    # model.add(Dense(total_words, activation='linear'))
    
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # model.compile(loss="mse", optimizer="rmsprop")
    history = model.fit(xs, ys, epochs=100, verbose=1)
    #print model.summary()
    print(model)
    
    # plot_graphs(history, 'accuracy')
    
    seed_text = "nồi áp suất"
    
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    print(seed_text)
    print(predicted)
    
test()

