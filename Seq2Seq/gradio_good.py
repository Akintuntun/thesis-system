import numpy as np
import time
import os
import torch
import gradio as gr
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from transformers import AutoModelForCausalLM, AutoTokenizer

# checkpoint 
checkpoint = "microsoft/DialoGPT-medium"
# download and cache tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# download and cache pre-trained model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Build a ChatBot class with all necessary modules to make a complete conversation
class ChatBot():
    # initialize
    def __init__(self):
        # once chat starts, the history will be stored for chat continuity
        self.chat_history_ids = None
        # make input ids global to use them anywhere within the object
        self.bot_input_ids = None
        # a flag to check whether to end the conversation
        self.end_chat = False
        
    def user_input(self, text):
        # preprocess input text
        # encode the new user input, add the eos_token and return a tensor in PyTorch
        self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, \
                                                   return_tensors='pt')

    def bot_response(self):
        # append the new user input tokens to the chat history
        # if chat has already begun
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1) 
        else:
            # if first entry, initialize bot_input_ids
            self.bot_input_ids = self.new_user_input_ids
        
        # define the new chat_history_ids based on the preceding chats
        # generated a response while limiting the total chat history to 1000 tokens, 
        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000, \
                                               pad_token_id=tokenizer.eos_token_id, temperature=0.7, repetition_penalty=1.3)
            
        # last ouput tokens from bot
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], \
                                    skip_special_tokens=True)

        # in case, bot fails to answer
        if response == "":
            response = self.random_response()
        return response
        
    # in case there is no response from model
    def random_response(self):
        i = -1
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
                                    skip_special_tokens=True)
        # iterate over history backwards to find the last token
        while response == '':
            i = i-1
            response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
                                        skip_special_tokens=True)
        # if it is a question, answer suitably
        if response.strip() == '?':
            reply = np.random.choice(["I don't know", 
                                     "I am not sure"])
        # not a question? answer suitably
        else:
            reply = np.random.choice(["Great", 
                                      "Fine. What's up?", 
                                      "Okay"])
        return reply

# Create an instance of the ChatBot class
chatbot = ChatBot()

# Define the function for Gradio to use
def chat_interface(text,history):
    chatbot.user_input(text)
    response = chatbot.bot_response()
    return response

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

allowed_paths = ["BackgroundMain.png"]

seafoam = Seafoam()

# Define the CSS for the ChatInterface
css ="""
    .gradio-container{
        background: url('file=BackgroundMain.png');
        background-size: cover;
        background-position: center;
    }
"""

# Create a ChatInterface with custom styling
chatbot_interface = gr.ChatInterface(
    fn=chat_interface,
    title="ChatBot",
    description="Chat with the ChatBot.",
    examples=[
        ["Hello!", ""],
        ["Who are you?", ""],
        ["What can you do?", ""],
        ["What do you do with my data?", ""]
    ],
    undo_btn=None,
    retry_btn=None,
    fill_height=True,
    theme=seafoam,
    css=css,
)

# Run the Gradio ChatInterface
chatbot_interface.launch(allowed_paths = allowed_paths, share=True)
