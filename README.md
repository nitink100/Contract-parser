# Legal Contract Parser

## Project Overview

This tool is a multi-layered solution for parsing legal contracts from PDF documents into a structured JSON format. It is engineered for a balance of accuracy and performance, using a fast, rule-based engine and an optional, advanced LLM (Large Language Model) enhancement for complex or non-standard documents.

The primary goal is to provide a reliable and extensible foundation for automating contract analysis, ensuring the output is always schema-compliant for seamless downstream integration.

## Key Features

* **Versatile Text Extraction**: Supports a wide range of PDFs, from native text documents to scanned images, by intelligently falling back from `pdfplumber` to `PyMuPDF` and `Tesseract OCR`.
* **Structured Parsing**: Automatically identifies and extracts critical contract metadata, including the title, contract type, and effective date.
* **Hierarchical Output**: Organizes content into a logical hierarchy of sections and clauses, complete with clear labels and indices.
* **LLM Integration (Optional)**: Provides a powerful enhancement for handling unstructured or highly complex legal language, ensuring high-quality output where traditional methods may fail.
* **Schema-Compliant Output**: All generated JSON output adheres strictly to a predefined `pydantic` data model, guaranteeing a predictable and reliable structure.

## Performance Note

This script is designed for a balance of speed and accuracy.

* **Rule-Based Parsing**: Typically fast, completing a 25-page PDF in approximately 5-15 seconds.
* **OCR**: CPU-intensive and can be slower, often taking 2-5 seconds per page for scanned documents.
* **LLM Enhancement**: Introduces network latency, with processing times of 5-15 seconds depending on the model and document size.

In most cases, especially for text-based PDFs, the target of under 60 seconds for a 25-page document is consistently met.

---

## Setup and Installation

### Prerequisites

1.  Python 3.8+
2.  Tesseract OCR Engine: Required for processing scanned PDFs. Ensure it is installed and added to your system's PATH.
    * macOS: `brew install tesseract`
    * Ubuntu: `sudo apt-get install tesseract-ocr`
    * Windows: Download from the [Tesseract GitHub Page](https://github.com/UB-Mannheim/tesseract/wiki/4.00-alpha-for-Windows).

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/contract-parser.git](https://github.com/your-username/contract-parser.git)
    cd contract-parser
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### API Key Configuration

To enable the LLM functionality, a Google Gemini API key is required. For security, this key must be set as an environment variable.

* macOS / Linux:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```
* Windows:
    ```bash
    set GEMINI_API_KEY="your_api_key_here"
    ```

---

## Usage

### Standard Parsing

For most documents, use the default rule-based parser.

```bash
python parser.py [input_pdf_path] [output_json_path]
```

### LLM-Forced Parsing

This option, which requires an API key, directs the script to bypass the rule-based parser and rely entirely on the LLM for document parsing. It is useful for highly complex or non-standard documents that may not conform to typical legal formats.

```bash
python parser.py [input_pdf_path] [output_json_path] --force-llm
```

### Example JSON output

{
  "title": "Service Agreement",
  "contract_type": "Service Agreement",
  "effective_date": "2025-09-10",
  "sections": [
    {
      "title": "Definitions",
      "number": "1",
      "clauses": [
        {
          "text": "This agreement defines certain terms...",
          "label": "1.1",
          "index": 0
        }
      ]
    }
  ]
}
