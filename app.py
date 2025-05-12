from transformers import pipeline
import gradio as gr

chatbot = pipeline("text-generation", model="distilgpt2", max_new_tokens=100)

def chat_with_bot(user_input):
    response = chatbot(user_input, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

interface = gr.Interface(fn=chat_with_bot, inputs="text", outputs="text", title="Simple Chatbot")
interface.launch()