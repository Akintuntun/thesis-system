import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load data from JSON file
with open('intentsV.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract patterns and tags from JSON data
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
X_seq = tokenizer.texts_to_sequences(patterns)

# Pad sequences to a fixed length
max_seq_length = 20  # Adjust as needed
X_pad = pad_sequences(X_seq, maxlen=max_seq_length)

# Encode target labels
label_to_index = {tag: i for i, tag in enumerate(sorted(set(tags)))}
y_encoded = np.array([label_to_index[tag] for tag in tags])

# Convert labels to one-hot encoding
y_one_hot = to_categorical(y_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_one_hot, test_size=0.2, random_state=42)

# Define BiLSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(len(label_to_index), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
report = classification_report(y_test_labels, y_pred_labels)

print(report)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Prediction
# Use the trained model to predict intents for user inputs
# Convert user inputs to sequences and pad them before prediction
# Then, decode the predicted labels back to their original classes using label_to_index
