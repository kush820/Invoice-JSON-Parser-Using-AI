import os
import uuid
import fitz  # PyMuPDF
import cv2
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import json
import google.generativeai as genai
from api_keys import genai_api_key  # Make sure you have Gemini SDK installed

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

BLOCKS_FOLDER = "extracted_blocks"
os.makedirs(BLOCKS_FOLDER, exist_ok=True)

genai.configure(api_key=genai_api_key)

def preprocess_page(page):
    """Convert PDF page to numpy image (BGR)."""
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return np_img

def process_image_lines(np_img, psm=7):
    """Extract text line-wise and crop line images."""
    custom_config = f'--oem 3 --psm {psm}'
    data = pytesseract.image_to_data(
        cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY),
        output_type=pytesseract.Output.DICT,
        config=custom_config
    )
    n_boxes = len(data['level'])
    page_lines = {}
    cropped_images = []

    for i in range(n_boxes):
        line_num = data['line_num'][i]
        left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text = data['text'][i].strip()
        if not text:
            continue

        page_lines.setdefault(f'line_{line_num}', {'coords': [], 'text': []})
        page_lines[f'line_{line_num}']['coords'].append((left, top, width, height))
        page_lines[f'line_{line_num}']['text'].append(text)

    structured_lines = {}
    for line_key, line in page_lines.items():
        structured_lines[line_key] = ' '.join(line['text'])

        # Crop image of the line
        coords = line['coords']
        x_min = min(c[0] for c in coords)
        y_min = min(c[1] for c in coords)
        x_max = max(c[0]+c[2] for c in coords)
        y_max = max(c[1]+c[3] for c in coords)
        crop_img = np_img[y_min:y_max, x_min:x_max]
        crop_filename = f"{uuid.uuid4()}.png"
        crop_path = os.path.join(BLOCKS_FOLDER, crop_filename)
        cv2.imwrite(crop_path, crop_img)
        cropped_images.append(crop_filename)

    # Combine all lines into one big text for LLM
    extracted_text = "\n".join([structured_lines[key] for key in sorted(structured_lines.keys())])
    return extracted_text, cropped_images, structured_lines

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
    all_cropped_images = []
    full_text = ""

    try:
        if ext == '.pdf':
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                np_img = preprocess_page(page)
                extracted_text, cropped_images, _ = process_image_lines(np_img, psm=7)
                full_text += f"\nPage {page_num+1}:\n{extracted_text}"
                all_cropped_images.extend(cropped_images)
        else:
            img = Image.open(filepath)
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            extracted_text, cropped_images, _ = process_image_lines(np_img, psm=7)
            full_text = extracted_text
            all_cropped_images.extend(cropped_images)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    # --- Gemini LLM processing ---
    try:
        with open("op_schema.json", "r") as f:
            schema = json.load(f)

        prompt = f"""
        You are an invoice data extraction assistant.
        Extract structured details from the following text.
        Return the output strictly in JSON format, following this schema:

        {json.dumps(schema, indent=2)}

        Invoice Text:
        {full_text}
        """

        # Correct usage
        response = genai.generate(
            model="gemini-1.5-flash",
            prompt=prompt,
            temperature=0
        )

        llm_output = response.result[0].content.strip()

        try:
            structured_json = json.loads(llm_output)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON from Gemini', 'raw_output': llm_output}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    return jsonify({
        'cropped_images': all_cropped_images,
        'structured_text': structured_json
    })


if __name__ == "__main__":
    app.run(debug=True)
