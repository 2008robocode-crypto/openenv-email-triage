import gradio as gr

def home():
    return "OpenEnv Email Triage Environment is Running ✅"

iface = gr.Interface(fn=home, inputs=[], outputs="text")

iface.launch(server_name="0.0.0.0", server_port=7860)