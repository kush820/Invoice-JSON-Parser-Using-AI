import os
import uuid
import fitz  # PyMuPDF
import cv2
import numpy as np
import json
# import google.generativeai as genai
from mistralai import Mistral
from api_keys import mistral_api_key
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import re

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

BLOCKS_FOLDER = "extracted_blocks"
os.makedirs(BLOCKS_FOLDER, exist_ok=True)
client = Mistral(api_key=mistral_api_key)


def preprocess_page(page):
    """Convert PDF page to numpy image (BGR)."""
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return np_img


def clean_json_output(raw_output):
    if not raw_output:
        return {}

    # Remove code fences
    raw_output = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw_output.strip())

    # Remove leading/trailing whitespace
    raw_output = raw_output.strip()

    # Replace escaped quotes and newlines
    raw_output = raw_output.replace('\\"', '').replace('\n', ' ').replace('\r', ' ')

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # As last resort, return raw string
        return {"raw_output": raw_output}


@app.route('/extract-blocks', methods=['POST'])
def extract_blocks():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ext = os.path.splitext(filename)[1].lower()
    cropped_images = []
    structured_text = {}
    full_text_list = []

    def process_image_blocks(np_img, page_num):
        data = pytesseract.image_to_data(cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY), output_type=pytesseract.Output.DICT)
        page_dict = {}
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            block_num = data['block_num'][i]
            par_num = data['par_num'][i]
            line_num = data['line_num'][i]
            left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i].strip()
            if not text:
                continue

            # Initialize hierarchy: block -> paragraph -> line
            page_dict.setdefault(f'block_{block_num}', {'coords': [], 'paragraphs': {}})
            block = page_dict[f'block_{block_num}']

            block['coords'].append((left, top, width, height))
            block['paragraphs'].setdefault(f'para_{par_num}', {})
            block['paragraphs'][f'para_{par_num}'].setdefault(f'line_{line_num}', [])
            block['paragraphs'][f'para_{par_num}'][f'line_{line_num}'].append(text)
            full_text_list.append(text)

        # Crop images and prepare structured text
        for block_key, block in page_dict.items():
            coords = block['coords']
            x_min = min(c[0] for c in coords)
            y_min = min(c[1] for c in coords)
            x_max = max(c[0]+c[2] for c in coords)
            y_max = max(c[1]+c[3] for c in coords)
            crop_img = np_img[y_min:y_max, x_min:x_max]
            crop_filename = f"{uuid.uuid4()}.png"
            crop_path = os.path.join(BLOCKS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop_img)
            cropped_images.append(crop_filename)

            structured_paragraphs = {}
            for para_key, lines in block['paragraphs'].items():
                paragraph_text = ' '.join([' '.join(lines[line]) for line in sorted(lines.keys())])
                structured_paragraphs[para_key] = paragraph_text

            structured_text[f'page_{page_num}_{block_key}'] = structured_paragraphs

    try:
        if ext == '.pdf':
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                np_img = preprocess_page(page)
                process_image_blocks(np_img, page_num)
            doc.close()
        else:
            img = Image.open(filepath).convert("RGB")
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            process_image_blocks(np_img, 0)
    finally:
        os.remove(filepath)

    full_text = "\n".join(full_text_list)
    # return jsonify({'structured_text': structured_text, 'full_text': full_text})

    # --- Gemini LLM processing ---
    try:
        with open("op_schema.json", "r") as f:
            schema = json.load(f)

        sys_prompt = f"""
        You are an invoice data extraction assistant.
        Task:
        Extract all relevant information from the invoice text below according to this exact JSON schema:
        {json.dumps(schema, indent=2)}
        Requirements:
        1. Return ONLY valid JSON. 
        2. Do NOT include markdown, backticks, code blocks, explanations, or extra text. 
        3. Do NOT add any comments or formatting. 
        4. The JSON must be compact and directly parsable with json.loads() in Python. 
        5. Clean the data: remove typos, extra spaces, newlines, or escape characters where possible. 
        6. Ensure all keys and values strictly follow the schema. If a value is missing, set it to null (not empty or "").
        """

        llm_input = json.dumps(structured_text, ensure_ascii=False)

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": f"Invoice Text:\n{llm_input}"
                }
            ]
        )

        # Mistral returns JSON object directly
        llm_output = response.choices[0].message.content

        # Remove Markdown code block if present
        structured_json = clean_json_output(llm_output)
        print(type(structured_json))
        json_res = json.dumps(structured_json, indent=2)
        print(json_res)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return app.response_class(json_res, mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True)
