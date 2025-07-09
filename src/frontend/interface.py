import gradio as gr
import requests



BACKEND_URL = "http://localhost:8000/extract"

# def process_card(image):
#     _, img_encoded = image.encode("png")
#     files = {"file": ("card.png", img_encoded, "image/png")}
#     response = requests.post(BACKEND_URL, files=files)
#     if response.status_code == 200:
#         data = response.json()
#         return data["structured_data"], data["ocr_text"]
#     else:
#         return {"error": "Failed to process image."}, ""

import io
import PIL.Image

def process_card(image: PIL.Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    files = {"file": ("card.png", buffer, "image/png")}
    
    response = requests.post(BACKEND_URL, files=files)

    if response.status_code == 200:
        data = response.json()

        return data["structured_data"], data["ocr_text"]
    else:
        return {"error": "Failed to process image."}, ""

with gr.Blocks() as demo:
    gr.Markdown("## Business Card Parser")

    with gr.Row():
        image_input = gr.Image(type="pil")
        result_json = gr.JSON(label="Parsed Contact Info")

    ocr_textbox = gr.Textbox(label="OCR Text")
    btn = gr.Button("Extract")

    btn.click(fn=process_card, inputs=[image_input], outputs=[result_json, ocr_textbox])

demo.launch()