import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import json
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

client = Mistral(api_key=mistral_api_key)


def preprocess_page(page):
    """Convert PDF page to numpy image (BGR)."""
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return np_img


def clean_json_output(raw_output):
    if not raw_output: return {}

    raw_output = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw_output.strip()) # Remove code fences
    raw_output = raw_output.strip().replace('\\"', '').replace('\n', ' ').replace('\r', ' ') # Remove leading/trailing whitespace & Replace escaped quotes and newlines

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # As last resort, return raw string
        return {"raw_output": raw_output}


@app.route('/invoice2json', methods=['POST'])
def extract_blocks():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ext = os.path.splitext(filename)[1].lower()
    structured_text = {}
    # full_text_list = []

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

            if not text: continue

            # Initialize hierarchy: block -> paragraph -> line
            page_dict.setdefault(f'block_{block_num}', {'coords': [], 'paragraphs': {}})
            block = page_dict[f'block_{block_num}']

            block['coords'].append((left, top, width, height))
            block['paragraphs'].setdefault(f'para_{par_num}', {})
            block['paragraphs'][f'para_{par_num}'].setdefault(f'line_{line_num}', [])
            block['paragraphs'][f'para_{par_num}'][f'line_{line_num}'].append(text)
            # full_text_list.append(text)

        # Prepare structured text (without saving crops)
        for block_key, block in page_dict.items():
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
                
                # Try direct text extraction first (for text-based PDFs)
                text = page.get_text("text").strip()
                
                if text:
                    # ✅ Text-based PDF — no OCR needed
                    structured_text[f"page_{page_num}"] = {"text": text}
                    # full_text_list.append(text)
                else:
                    # 🖼️ Image-based PDF — fallback to OCR
                    np_img = preprocess_page(page)
                    process_image_blocks(np_img, page_num)
            
            doc.close()

        else:
            img = Image.open(filepath).convert("RGB")
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            process_image_blocks(np_img, 0)
    finally:
        os.remove(filepath)

    # full_text = "\n".join(full_text_list)
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

        # llm_input = json.dumps(structured_text, ensure_ascii=False)

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": f"Invoice Text:\n{json.dumps(structured_text, ensure_ascii=False)}"
                }
            ]
        )

        # Mistral returns JSON object directly
        llm_output = response.choices[0].message.content

        # Remove Markdown code block if present
        structured_json = clean_json_output(llm_output)
        print(type(structured_json))
        json_res = json.dumps(structured_json, indent=2) # Fixing the JSON arrangement
        # print(json_res)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return app.response_class(json_res, mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True)
