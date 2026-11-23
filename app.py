"""Gradio demo for the Central Bank Communications Assistant (placeholder).

The real version will call the RAG pipeline (rag/) and the tone classifier (classifier/).
"""
import gradio as gr


def answer(question: str, bank: str) -> str:
    return f"(placeholder) You asked about {bank}: {question}"


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Dropdown(["All", "Fed", "ECB", "BoE", "BoJ"], value="All", label="Central bank"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Central Bank Communications Assistant",
)

if __name__ == "__main__":
    demo.launch()
