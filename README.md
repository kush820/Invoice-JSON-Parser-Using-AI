# Invoice-JSON-Parser-Using-AI

A small toolkit that demonstrates converting invoices (PDF / images) to structured JSON using AI model calls.

Overview
--------

This repository contains example scripts that call external AI services to extract structured information from invoices and output a JSON file. The code relies on API keys for the AI providers — those keys must be provided locally before running the scripts.

Required files
--------------

- `api_keys.py` — where you should put your API keys (see next section).
- `requirements.txt` — Python dependencies used by the scripts.
- `invoice_to_json_v2.py` — main script to run (entry point).

Quickstart
----------

1) Set API keys

   - Open `api_keys.py` and set the API key variables. Example shape:

   ```python
   genai_api_key = "<enter your genai api key here>"
   google_ai_studio_api_key = "<enter your google ai studio api key here>"
   mistral_api_key = "<enter your mistral api key here>"
   ```

   - Do NOT commit real keys to the repository. Treat keys like secrets.

   Alternative (recommended for production): export keys as environment variables and update the code to read from `os.environ` instead of `api_keys.py`.
2) Create and activate a virtual environment

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   On Windows (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3) Run the main script

   With the virtual environment active:

   ```bash
   python invoice_to_json_v2.py
   ```

   The script will attempt to parse `invoice.png` (included for demo) or PDFs present in the repository and will write the parsed output to `parsed_output.json` on success. If API keys are missing or invalid, the script will print authentication errors.

Output
------

- `parsed_output.json` — example output produced by the script when parsing succeeds.

Security notes
--------------

- Never commit `api_keys.py` containing real keys. Use `.gitignore` or environment variables for secrets.
- Rotate keys if they are accidentally committed.

Troubleshooting
---------------

- Missing dependencies: re-run `pip install -r requirements.txt` and inspect the error messages.
- Authentication errors: verify key values in `api_keys.py` or your environment variables.
- If you want the project to use environment variables instead of `api_keys.py`, I can provide a small patch to read keys from `os.environ`.
