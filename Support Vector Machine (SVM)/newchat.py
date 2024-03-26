import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

with open('intentsV.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df = pd.DataFrame.from_dict(dic)
print (df['tag'].unique())
# Distribution of Intents
intent_counts = df['tag'].value_counts()
plt.bar(intent_counts.index, intent_counts.values)
plt.title('Distribution of Intents')
plt.xlabel('Intents')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pattern and Response Analysis
df['pattern_count'] = df['patterns'].apply(lambda x: len(x))
df['response_count'] = df['responses'].apply(lambda x: len(x))
avg_pattern_count = df.groupby('tag')['pattern_count'].mean()
avg_response_count = df.groupby('tag')['response_count'].mean()

plt.bar(avg_pattern_count.index, avg_pattern_count.values, label='Average Pattern Count')
plt.bar(avg_response_count.index, avg_response_count.values, label='Average Response Count')
plt.title('Pattern and Response Analysis')
plt.xlabel('Intents')
plt.ylabel('Average Count')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Split the dataset into training and testing sets
X = df['patterns']
y = df['tag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
model = SVC()
model.fit(X_train_vec, y_train)

# Predict intents for the testing set
y_pred = model.predict(X_test_vec)

# Evaluate the model's performance
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

print(classification_report(y_test, y_pred))

# Convert float values in the report to dictionaries
report = {label: {metric: report[label][metric] for metric in report[label]} for label in report if isinstance(report[label], dict)}

# Extract evaluation metrics
labels = list(report.keys())
evaluation_metrics = ['precision', 'recall', 'f1-score']
metric_scores = {metric: [report[label][metric] for label in labels if label in report] for metric in evaluation_metrics}

# Intent Prediction Model Performance
fig, ax = plt.subplots()
for metric in evaluation_metrics:
    ax.bar(labels, metric_scores[metric], label=metric)
ax.set_title('Intent Prediction Model Performance')
ax.set_xlabel('Intent')
ax.set_ylabel('Score')
ax.legend()
plt.xticks(rotation=45)
plt.show()

# Prediction Model Deployment

# A trained SVM model named 'model' and a vectorizer named 'vectorizer'



# Function to predict intents based on user input
def predict_intent(user_input):
    # Vectorize the user input
    user_input_vec = vectorizer.transform([user_input])

    # Predict the intent
    intent = model.predict(user_input_vec)[0]

    return intent

# Create a response dictionary dynamically from the intents dataset
response_dict = {}
for intent_entry in data['intents']:
    intent = intent_entry['tag']
    responses = intent_entry['responses']
    # You can choose a response randomly or use a specific strategy here
    response = responses[0]  # For simplicity, using the first response
    response_dict[intent] = response

# Define the generate_response function using the response dictionary
def generate_response(intent):
    # Check if the intent exists in the response dictionary
    if intent in response_dict:
        response = response_dict[intent]
    else:
        response = "I'm here to help. Please let me know how I can assist you."
    return response

# Example usage
while True:
    # Get user input
    user_input = input("User: ")

    if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
    # Predict intent
    intent = predict_intent(user_input)

    # Generate response
    response = generate_response(intent)

    print("Chatbot:", response)
