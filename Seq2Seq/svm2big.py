import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
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

# Split the dataset into features (X) and target labels (y)
X_train = data_train['text']
y_train = data_train['label']
X_test = data_test['text']
y_test = data_test['label']
X_val = data_val['text']
y_val = data_val['label']

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_val_vec = vectorizer.transform(X_val)

# Train a Support Vector Machine (SVM) classifier
model = SVC()
model.fit(X_train_vec, y_train)

# Predict intents for the testing set
y_pred = model.predict(X_test_vec)

# Evaluate the model's performance
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Function to predict intents based on user input
def predict_intent(user_input):
    # Vectorize the user input
    user_input_vec = vectorizer.transform([user_input])

    # Predict the intent
    intent = model.predict(user_input_vec)[0]

    return intent

# Function to generate responses based on predicted intents
def generate_response(intent):
    # Implement your logic here to generate appropriate responses based on the predicted intents
    if intent == 'greeting':
        response = "Hello! How can I assist you today?"
    elif intent == 'farewell':
        response = "Goodbye! Take care."
    elif intent == 'question':
        response = "I'm sorry, I don't have the information you're looking for."
    else:
        response = "I'm here to help. Please let me know how I can assist you."

    return response

# Function to plot distribution of intents
def plot_intent_distribution(dataframe):
    intent_counts = dataframe['label'].value_counts()
    fig = go.Figure(data=[go.Bar(x=intent_counts.index, y=intent_counts.values)])
    fig.update_layout(title='Distribution of Intents', xaxis_title='Intents', yaxis_title='Count')
    fig.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    fig = go.Figure(data=go.Heatmap(z=cm_df.values, x=labels, y=labels, colorscale='Blues'))
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='True')
    fig.show()

# Function to plot prediction model performance
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

# Function to plot pattern and response analysis

plot_intent_distribution(data_train)

plot_confusion_matrix(y_test, y_pred, labels=model.classes_)

plot_model_performance(report)

# Total accuracy
total_accuracy = report['accuracy']

# Overall precision, recall, and F1-score
overall_precision = report['macro avg']['precision']
overall_recall = report['macro avg']['recall']
overall_f1_score = report['macro avg']['f1-score']

print("Total Accuracy:", total_accuracy)
print("Overall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F1-score:", overall_f1_score)
