import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import json
from mistralai import Mistral
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import re

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
load_dotenv()
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = Mistral(api_key=os.getenv("mistral_api_key"))


def preprocess_page(page):
    """Convert PDF page to numpy image (BGR)."""
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return np_img


def clean_json_output(raw_output):
    """Clean LLM output and parse as JSON while preserving structure."""
    if not raw_output:
        return {}

    # Remove code fences and extra escapes
    raw_output = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw_output.strip())
    raw_output = raw_output.strip().replace('\\"', '').replace('\r', ' ')
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"raw_output": raw_output}


@app.route('/invoice2json', methods=['POST'])
def extract_blocks():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the client wants a downloadable file
    toSave = request.form.get('save', 'true').lower() == 'true'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ext = os.path.splitext(filename)[1].lower()
    structured_text = {}

    def process_image_blocks(np_img, page_num):
        data = pytesseract.image_to_data(
            cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY),
            output_type=pytesseract.Output.DICT
        )
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

            page_dict.setdefault(f'block_{block_num}', {'coords': [], 'paragraphs': {}})
            block = page_dict[f'block_{block_num}']
            block['coords'].append((left, top, width, height))
            block['paragraphs'].setdefault(f'para_{par_num}', {})
            block['paragraphs'][f'para_{par_num}'].setdefault(f'line_{line_num}', [])
            block['paragraphs'][f'para_{par_num}'][f'line_{line_num}'].append(text)

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
                text = page.get_text("text").strip()

                if text:
                    structured_text[f"page_{page_num}"] = {"text": text}
                else:
                    np_img = preprocess_page(page)
                    process_image_blocks(np_img, page_num)
            doc.close()
        else:
            img = Image.open(filepath).convert("RGB")
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            process_image_blocks(np_img, 0)
    finally:
        os.remove(filepath)

    # --- LLM Processing ---
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
        6. Ensure all keys and values strictly follow the schema. If a value is missing, set it to null.
        """

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Invoice Text:\n{json.dumps(structured_text, ensure_ascii=False)}"}
            ]
        )

        llm_output = response.choices[0].message.content
        structured_json = clean_json_output(llm_output)
        json_res = json.dumps(structured_json, indent=2)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    result = {"structured_json": json_res}

    # --- Save JSON file if requested ---
    if toSave:
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(llm_output)  # save exact LLM output to preserve key order

        result["download_file"] = output_path  # return path for Streamlit download

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
