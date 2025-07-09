# # backend/main.py
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from backend.ocr import extract_text_from_image
# from backend.llm_parser import generate_structured_data
# from backend.storage import save_to_csv

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/extract")
# async def extract(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     ocr_text = extract_text_from_image(image_bytes)
#     structured_data = generate_structured_data(ocr_text)
#     save_to_csv(structured_data)
#     return {"ocr_text": ocr_text, "structured_data": structured_data}

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

from PIL import Image
import io
import requests
import os
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURATION ===
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("❌ Mistral API key is missing. Please set MISTRAL_API_KEY in your .env file.")

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}
CSV_PATH = "results/contacts.csv"

os.makedirs("results", exist_ok=True)

# Ensure CSV file exists
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["name", "title", "company", "email", "phone", "website", "address"]).to_csv(CSV_PATH, index=False)

# === UTILITY FUNCTIONS ===
import re
import json

def extract_json_from_response(response_text):
    # Extract JSON inside ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        raise ValueError("❌ Could not extract JSON from LLM response.")

def extract_text_from_image(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text.strip()

# def generate_structured_data(ocr_text: str) -> dict:
#     prompt = f"""
#     Extract the following fields from this business card:
#     - Name
#     - Job Title
#     - Company
#     - Email
#     - Phone
#     - Website
#     - Address

#     Return output in JSON format.

#     OCR Extracted Text:
#     {ocr_text}
#     """
#     body = {
#         "model": "mistral-tiny",
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.3
#     }
#     response = requests.post(MISTRAL_API_URL, json=body, headers=HEADERS)
#     result = response.json()
#     return eval(result["choices"][0]["message"]["content"])

def generate_structured_data(ocr_text: str) -> dict:
    prompt = f"""
Extract the following fields from this business card:
- Name
- Job Title
- Company
- Email
- Phone
- Website
- Address

Return output in JSON format.

Return only the structured data as a JSON object. Do not include any explanation or text — only return valid JSON.

OCR Extracted Text:
\"\"\"
{ocr_text}
\"\"\"
"""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-tiny",  # Replace with the correct model if needed
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts structured contact data from unstructured text."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    # # 🔍 Log full result for debugging
    # print("🧾 Mistral API response:", result) # For, testing

    # if "choices" not in result:
    #     raise ValueError(f"Mistral API Error: {result}")

    # return eval(result["choices"][0]["message"]["content"])

    content = result["choices"][0]["message"]["content"]
    # structured_data = extract_json_from_response(content)
    structured_data = json.loads(content)
    return structured_data

def save_to_csv(parsed_data: dict):
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([parsed_data])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

# === FASTAPI SETUP ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # image_bytes = await file.read()
    # ocr_text = extract_text_from_image(image_bytes)
    # structured_data = generate_structured_data(ocr_text)
    # save_to_csv(structured_data)
    # return {"ocr_text": ocr_text, "structured_data": structured_data}

    print("📥 Reading image...")
    image_bytes = await file.read()

    print("🧾 Running OCR...")
    ocr_text = extract_text_from_image(image_bytes)
    print("📄 OCR Text:", ocr_text[:100])  # Preview first 100 chars

    print("🤖 Calling LLM...")
    structured_data = generate_structured_data(ocr_text)
    print("✅ LLM Response:", structured_data)

    print("💾 Saving to CSV...")
    save_to_csv(structured_data)
    print("✅ Done")

    return {"ocr_text": ocr_text, "structured_data": structured_data}