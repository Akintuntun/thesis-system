import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable
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
with open('intentsV.json', 'r', encoding='utf-8') as intents_file:
    intents_data = json.load(intents_file)

# Specify allowed file paths
allowed_paths = ["BackgroundMain.png"]

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )


seafoam = Seafoam()

css ="""
    .gradio-container{
        background: url('file=BackgroundMain.png');
        background-size: cover;
        background-position: center;
    }
"""

with gr.Blocks(css=css, theme=seafoam) as demo:
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    
    def predict_intent(text):
        input_sequence = tokenizer.texts_to_sequences([text])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=model.input_shape[1])
        predictions = model.predict(padded_input_sequence)
        predicted_intent = label_encoder.inverse_transform([np.argmax(predictions)])
        return predicted_intent[0]

    def get_response(intent):
        for intent_data in intents_data['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                return np.random.choice(responses)
        return "I'm not sure how to respond to that."


    # Function to process user input and generate chatbot response
    def chatbot_response(user_message, history):
        # Get the predicted intent
        predicted_intent = predict_intent(user_message)
        # Get the response based on the intent
        response = get_response(predicted_intent)

        history.append((user_message, response))

        return response, history


    msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()

	