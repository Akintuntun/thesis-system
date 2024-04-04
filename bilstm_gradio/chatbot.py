from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import numpy as np
# Load the trained BiLSTM model
model = load_model('bilstm_chatbot_model.h5')


# Load the Tokenizer and Label Encoder used during training
with open('tokenizer_and_encoder.pkl', 'rb') as file:
    data = pickle.load(file)
    tokenizer = data['tokenizer']
    label_encoder = data['encoder']

# Load intents data from intents.json
with open('intentsE.json', 'r', encoding='utf-8') as intents_file:
    intents_data = json.load(intents_file)

# Function to get model prediction
def predict_intent(text):
    input_sequence = tokenizer.texts_to_sequences([text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=model.input_shape[1])
    predictions = model.predict(padded_input_sequence)
    predicted_intent = label_encoder.inverse_transform([np.argmax(predictions)])
    return predicted_intent[0]

# Function to get response based on intent
def get_response(intent):
    for intent_data in intents_data['intents']:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
            return np.random.choice(responses)
    return "I'm not sure how to respond to that."

# Simple chat loop
print("Chatbot: Hello! Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    # Get the predicted intent
    predicted_intent = predict_intent(user_input)
    
    # Get the response based on the intent
    response = get_response(predicted_intent)
    
    print(f"Chatbot: {response}")

