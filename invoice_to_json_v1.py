import os
import fitz
import docx
import pytesseract
from PIL import Image
import cv2
import numpy as np
import uuid
import json
import google.generativeai as genai
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import easyocr
from api_keys import genai_api_key

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
genai.configure(genai_api_key)




def process_document(file_path, lang='eng'):
    ext = os.path.splitext(file_path)[1].lower()
    # if ext == '.docx':
    #     return process_docx(file_path)
    if ext == '.pdf':
        return process_pdf(file_path, lang)
    elif ext in ['.jpeg', '.jpg', '.png']:
        return process_image(file_path, lang)
    else:
        return "Unsupported file type"

# def process_docx(docx_file):
#     doc = docx.Document(docx_file)
#     return '\n'.join([para.text for para in doc.paragraphs])

def process_pdf(pdf_file, lang='eng'):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text if text.strip() else perform_ocr_on_pdf(pdf_file)
    # return text if text.strip() else perform_ocr_on_pdf_easyocr(pdf_file)
    # return text if text.strip() else perform_ocr_on_pdf_kerasocr(pdf_file)

def perform_ocr_on_pdf(pdf_file, lang='eng'):
    """Perform OCR on image-based PDFs, block by block."""
    text = ""
    with fitz.open(pdf_file) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            processed_img = preprocess_image_for_ocr(img)

            # 🔹 Save processed image for analysis
            save_path = f"processedImages/processed_page_{page_num+1}.png"
            processed_img.save(save_path, quality=95, dpi=(300, 300))
            print(f"[Saved] {save_path}")

            # Extract OCR per block with --psm 3
            text += pytesseract.image_to_string(
                processed_img, lang=lang, config='--psm 3'
            )
    return text

# ===================================================================================================================================

reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if GPU is available

def perform_ocr_on_pdf_easyocr(pdf_file):
    """Perform OCR on image-based PDFs using EasyOCR, block by block."""
    text = ""
    
    # Ensure folder exists
    os.makedirs("processedImages", exist_ok=True)

    with fitz.open(pdf_file) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Preprocess image (same as before)
            processed_img = preprocess_image_for_ocr(img)

            # Save processed image
            save_path = f"processedImages/processed_page_{page_num+1}.png"
            processed_img.save(save_path, quality=95, dpi=(300, 300))
            print(f"[Saved] {save_path}")

            # EasyOCR expects numpy array
            np_img = np.array(processed_img)
            result = reader.readtext(np_img)  # returns list of (bbox, text, confidence)
            
            # Combine extracted text
            page_text = "\n".join([r[1] for r in result])
            text += page_text + "\n"
    
    return text


# ============================================================================================

# pipeline = keras_ocr.pipeline.Pipeline()

# def perform_ocr_on_pdf_kerasocr(pdf_file):
#     """Perform OCR on image-based PDFs using KerasOCR, block by block."""
#     text = ""
    
#     # Ensure folder exists
#     os.makedirs("processedImages", exist_ok=True)

#     with fitz.open(pdf_file) as doc:
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

#             # Preprocess image
#             processed_img = preprocess_image_for_ocr(img)

#             # Save processed image
#             save_path = f"processedImages/processed_page_{page_num+1}.png"
#             processed_img.save(save_path, quality=95, dpi=(300, 300))
#             print(f"[Saved] {save_path}")

#             # KerasOCR expects numpy array
#             np_img = np.array(processed_img)
#             prediction_groups = pipeline.recognize([np_img])  # returns list of predictions

#             # Flatten predictions and extract text
#             page_text = "\n".join([text for box, text in prediction_groups[0]])
#             text += page_text + "\n"

#     return text



def process_image(image_file, lang='eng'):
    """Perform OCR on images, block by block."""
    img = Image.open(image_file)
    processed_img = preprocess_image_for_ocr(img)

    # Extract OCR per block with --psm 3
    return pytesseract.image_to_string(
        processed_img, lang=lang, config='--psm 3'
    )


def preprocess_image_for_ocr(img):
    """Convert image to pure black and white for max clarity."""
    # 1. Convert to grayscale for simplified processing
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 3. Sharpen the image to make text crisp
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpening_kernel)

    return Image.fromarray(sharpened)




@app.route('/parse-invoice', methods=['POST'])
def parse_invoice():
    """Upload file, extract text, and parse invoice using Gemini."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    lang = request.form.get('lang', 'eng')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    
    extracted_text = process_document(filepath)
    os.remove(filepath)  # Clean up after processing
    # return extracted_text


    with open("op_schema.json", "r") as f: schema = json.load(f) # Step 2: Load the schema

    # Step 3: Prepare prompt
    prompt = f"""
    You are an invoice data extraction assistant.
    Extract structured details from the following text.
    Return the output strictly in JSON format, following this schema:

    {json.dumps(schema, indent=2)}

    Invoice Text:
    {extracted_text}
    """

    # Step 4: Call Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    llm_output = response.text.strip()

    # Step 5: Ensure valid JSON
    try:
        structured_json = json.loads(llm_output)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON from Gemini', 'raw_output': llm_output}), 500

    print(jsonify(structured_json))

    return jsonify(structured_json)


if __name__ == '__main__':
    app.run(debug=True)
