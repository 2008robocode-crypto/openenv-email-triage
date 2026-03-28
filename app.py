import gradio as gr

def home():
    return "OpenEnv Email Triage Environment is Running ✅"

iface = gr.Interface(fn=home, inputs=[], outputs="text")

if __name__ == "__main__":
    iface.launch()