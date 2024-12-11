
#Name:P.yagnesh reddy
# **Roll number:HU21CSEN0100681**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
#Enter The link where the data set is present.
file_path = '/content/drive/MyDrive/datastu1.csv'
df = pd.read_csv(file_path)

# Assuming 'description' is the column containing textual data
text_data = df['description'].astype(str).values
print(df.head(20))
# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Convert y to categorical if you have more than two classes
# y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_length-1))
lstm_model.add(LSTM(units=100))
lstm_model.add(Dense(units=total_words, activation='softmax'))

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model
history_lstm = lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Build the RNN model
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_length-1))
rnn_model.add(SimpleRNN(units=100))
rnn_model.add(Dense(units=total_words, activation='softmax'))

# Compile the RNN model
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the RNN model
history_rnn = rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Plot training history for LSTM model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='LSTM Training Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Validation Accuracy')
plt.title('LSTM Model Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training history for RNN model
plt.subplot(1, 2, 2)
plt.plot(history_rnn.history['accuracy'], label='RNN Training Accuracy')
plt.plot(history_rnn.history['val_accuracy'], label='RNN Validation Accuracy')
plt.title('RNN Model Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate models on the test set
lstm_test_loss, lstm_test_accuracy = lstm_model.evaluate(X_test, y_test)
print(f'LSTM Test Accuracy: {lstm_test_accuracy}')

rnn_test_loss, rnn_test_accuracy = rnn_model.evaluate(X_test, y_test)
print(f'RNN Test Accuracy: {rnn_test_accuracy}')
rnn_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)