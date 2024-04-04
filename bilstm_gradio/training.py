import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
import json
import pickle

# Load intents data from intents.json
with open('intentsE.json', 'r', encoding='utf-8') as intents_file:
    intents_data = json.load(intents_file)

# Extract patterns and associated intents from intents data
patterns = []
intents = []

for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intents.append(intent['tag'])

# Print information about the dataset
print(len(patterns), "patterns")
print(len(set(intents)), "unique intents")

# Tokenize the patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(patterns)

# Pad sequences for consistent input size
padded_sequences = pad_sequences(sequences)

# Encode intents using LabelEncoder
label_encoder = LabelEncoder()
encoded_intents = label_encoder.fit_transform(intents)

# Create input-output pairs
X = padded_sequences
y = encoded_intents

# Save the Tokenizer and Label Encoder
with open('tokenizer_and_encoder.pkl', 'wb') as file:
    pickle.dump({'tokenizer': tokenizer, 'encoder': label_encoder}, file)

# Define the BiLSTM model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=X.shape[1]))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))

# Print information about the model
print("Total words in the tokenizer:", total_words)
print("Model summary:")
model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=80, batch_size=8)

# Save the model
model.save('bilstm_chatbot_model.h5')

# Plot the training progress
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper left')

plt.tight_layout()
plt.show()
