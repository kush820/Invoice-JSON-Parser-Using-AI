import streamlit as st
import requests
import json
from io import BytesIO

st.set_page_config(page_title="Invoice to JSON", layout="centered")

st.title("Invoice to JSON Converter")

# --- SECTION 1: Inputs ---
st.header("Step 1: Upload Files")

col1, col2 = st.columns(2)

with col1:
    schema_file = st.file_uploader("Upload JSON Schema", type=["json"])

with col2:
    invoice_file = st.file_uploader("Upload Invoice (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

downloadable = st.checkbox("Downloadable File")

process_btn = st.button("Process Invoice")

# --- SECTION 2: Output ---
st.header("Step 2: Output")

if process_btn:
    if not schema_file or not invoice_file:
        st.warning("Please upload both the JSON schema and the invoice file.")
    else:
        # Save uploaded files to temporary in-memory files
        schema_content = schema_file.read()
        invoice_content = invoice_file.read()

        # Prepare files for POST request
        files = {
            "file": (invoice_file.name, invoice_content),
        }

        # Send request to Flask API
        try:
            response = requests.post("http://127.0.0.1:5000/invoice2json", files=files)
            if response.status_code == 200:
                result = response.json()
                structured_json = result.get("structured_json", {})
                
                # Provide download if checkbox selected
                if downloadable:
                    # Fetch file from Flask download endpoint
                    download_link = f"http://127.0.0.1:5000/download/{invoice_file.name.split('.')[0]}.json"
                    st.markdown(f"[Download JSON File]({download_link})", unsafe_allow_html=True)

                # Display JSON output
                st.subheader("Processed JSON Output")
                st.json(structured_json)


            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to the Flask server: {e}")
