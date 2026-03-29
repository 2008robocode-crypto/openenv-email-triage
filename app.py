import gradio as gr

def home():
    return "OpenEnv Email Triage Environment is Running ✅"

demo = gr.Interface(fn=home, inputs=[], outputs="text")

demo.launch()