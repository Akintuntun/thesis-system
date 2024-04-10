import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Load data from is_train.json, is_test.json, and is_val.json
train_data = json.load(open('is_train.json', 'r', encoding='utf-8'))
test_data = json.load(open('is_test.json', 'r', encoding='utf-8'))
val_data = json.load(open('is_val.json', 'r', encoding='utf-8'))

# Convert data into Pandas DataFrames
train = np.array(train_data)
val = np.array(val_data)
test = np.array(test_data)

data_train = pd.DataFrame({
    'text': train[:, 0],
    'label': train[:, 1]
})
data_val = pd.DataFrame({
    'text': val[:, 0],
    'label': val[:, 1]
})
data_test = pd.DataFrame({
    'text': test[:, 0],
    'label': test[:, 1]
})

# Convert string labels to numerical indices
label_to_index = {label: index for index, label in enumerate(data_train['label'].unique())}
data_train['label'] = data_train['label'].map(label_to_index)
data_val['label'] = data_val['label'].map(label_to_index)
data_test['label'] = data_test['label'].map(label_to_index)


# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_train['text'])

X_train_seq = tokenizer.texts_to_sequences(data_train['text'])
X_val_seq = tokenizer.texts_to_sequences(data_val['text'])
X_test_seq = tokenizer.texts_to_sequences(data_test['text'])

# Pad sequences for consistent input size
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_val_padded = pad_sequences(X_val_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Define the BiLSTM model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(len(data_train['label'].unique()), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, data_train['label'], epochs=10, batch_size=32, validation_data=(X_val_padded, data_val['label']))

# Evaluate the model on the test set
# Predict intents for the testing set
y_pred_probabilities = model.predict(X_test_padded)
y_pred = np.argmax(y_pred_probabilities, axis=1)

report = classification_report(data_test['label'], y_pred, output_dict=True, zero_division=0)


# Function to plot distribution of intents
def plot_intent_distribution(dataframe):
    intent_counts = dataframe['label'].value_counts()
    fig = go.Figure(data=[go.Bar(x=intent_counts.index, y=intent_counts.values)])
    fig.update_layout(title='Distribution of Intents', xaxis_title='Intents', yaxis_title='Count')
    fig.show()


# Plot prediction model performance
def plot_model_performance(report):
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    for intent, scores in report.items():
        if intent != 'accuracy':
            data.append([intent] + [scores[metric] for metric in metrics])

    df = pd.DataFrame(data, columns=['Intent'] + metrics)
    fig = go.Figure(data=[go.Bar(name=metric, x=df['Intent'], y=df[metric]) for metric in metrics])
    fig.update_layout(title='Prediction Model Performance', xaxis_title='Intent', yaxis_title='Score', barmode='group')
    fig.show()


plot_intent_distribution(data_train)


plot_model_performance(report)

# Calculate total accuracy, precision, recall, and f1-score from the classification report
accuracy = report['accuracy']
precision = report['macro avg']['precision']
recall = report['macro avg']['recall']
f1 = report['macro avg']['f1-score']

# Print the metrics
print("Total Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
