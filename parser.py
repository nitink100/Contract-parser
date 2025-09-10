# parser.py
#
# First Name: Nitin
# Last Name: Kanna
#
# PERFORMANCE NOTE:
# This script is designed for a balance of accuracy and performance.
# - Rule-based parsing is fast (~5-15 seconds for a 25-page PDF).
# - OCR for scanned PDFs is CPU-intensive and can be slow (~2-5 seconds per page).
# - Optional LLM enhancement adds network latency (~5-15 seconds depending on the model and document size).
# The target of <= 60s for a 25-page PDF should be met in most cases, especially for text-based PDFs.

# --- SETUP INSTRUCTIONS ---
#
# 1. Install required Python packages:
#    pip install -r requirements.txt
#
# 2. Install Google's Tesseract OCR Engine (required for scanned PDFs):
#    - macOS: brew install tesseract
#    - Ubuntu: sudo apt-get install tesseract-ocr
#    - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
#    Ensure the 'tesseract' command is in your system's PATH.
#
# 3. (Optional) Set your API Key for LLM Enhancement:
#    The script requires the GEMINI_API_KEY environment variable to be set.
#    - Windows: set GEMINI_API_KEY="your_api_key_here"
#
# --- EXECUTION ---
#
# Basic command:
# python parser.py /path/to/your/contract.pdf /path/to/your/output.json
#
# Forcing LLM Parsing:
# To skip the rule-based parser and use the LLM for the entire document,
# add the --force-llm flag. This is useful for complex or non-standard
# documents. Note: This requires a valid API key to be set.
#
# Example with flag:
# python parser.py /path/to/complex.pdf /path/to/output.json --force-llm
#
# -----------------------------------------------------------------------------

import sys
import os
import json
import re
import logging
import argparse
from datetime import datetime
from typing import List, Optional, Union

# Third-party imports
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# 1. JSON OUTPUT SCHEMA (using Pydantic for validation)
# =============================================================================

class ClauseModel(BaseModel):
    """Schema for a single clause."""
    text: str
    label: str = ""  # Default to empty string as per rules
    index: int

class SectionModel(BaseModel):
    """Schema for a contract section."""
    title: str
    number: Optional[str] = None # Default to None (becomes null in JSON)
    clauses: List[ClauseModel]

class ContractModel(BaseModel):
    """Top-level schema for the entire contract."""
    title: str
    contract_type: str
    effective_date: Optional[str] = None # YYYY-MM-DD or null
    sections: List[SectionModel]

# =============================================================================
# 2. LAYER 1: TEXT EXTRACTION ENGINE
# =============================================================================

def extract_text_with_pdfplumber(pdf_path: str) -> Optional[str]:
    """Extracts text using pdfplumber, good for text-based PDFs."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        logging.warning(f"pdfplumber failed: {e}")
        return None

def extract_text_with_pymupdf(pdf_path: str) -> Optional[str]:
    """Fallback text extraction using PyMuPDF, also good for text-based."""
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logging.warning(f"PyMuPDF failed: {e}")
        return None

def extract_text_with_ocr(pdf_path: str) -> Optional[str]:
    """Last resort OCR extraction for scanned/image-based PDFs."""
    try:
        full_text = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
                full_text.append(page_text)
        return "\n".join(full_text)
    except Exception as e:
        logging.error(f"OCR extraction failed: {e}. Is Tesseract installed and in PATH?")
        return None

def get_pdf_text(pdf_path: str) -> str:
    """Orchestrates text extraction with fallbacks."""
    logging.info(f"Starting text extraction for {pdf_path}")

    if not os.path.exists(pdf_path):
        logging.error(f"Input file not found at path: {pdf_path}")
        return ""
    
    text = extract_text_with_pdfplumber(pdf_path)
    if text and text.strip():
        logging.info("Text extracted successfully with pdfplumber.")
        return text

    text = extract_text_with_pymupdf(pdf_path)
    if text and text.strip():
        logging.info("Text extracted successfully with PyMuPDF as a fallback.")
        return text

    logging.warning("Primary methods failed. Attempting OCR... (this may be slow)")
    text = extract_text_with_ocr(pdf_path)
    if text and text.strip():
        logging.info("Text extracted successfully with OCR.")
        return text

    logging.error("All text extraction methods failed.")
    return ""

# =============================================================================
# 3. LAYER 2-4: RULE-BASED PARSING ENGINE
# =============================================================================

class RuleBasedParser:
    """A rule-based engine to parse text into the contract schema."""

    def __init__(self, text: str):
        self.text = text
        self.lines = [line.strip() for line in text.split('\n') if line.strip()]

    def parse(self) -> ContractModel:
        """Main parsing logic."""
        metadata = self._extract_metadata()
        sections = self._segment_sections_and_clauses()

        # If no sections were found, create a default one
        if not sections:
            sections = [
                SectionModel(
                    title="Contract Content",
                    number=None,
                    clauses=[ClauseModel(text=" ".join(self.lines), label="", index=0)]
                )
            ]

        return ContractModel(
            title=metadata.get("title", "Contract"),
            contract_type=metadata.get("contract_type", "Agreement"),
            effective_date=metadata.get("effective_date"),
            sections=sections
        )

    def _extract_metadata(self) -> dict:
        """Extracts title, type, and effective date from the full text."""
        # Title Extraction (heuristic: first few, short, all-caps lines)
        title = "Contract"
        for line in self.lines[:10]:
            if len(line.split()) < 8 and line.isupper():
                title = line.title()
                break

        # Contract Type Extraction (heuristic: keywords in title)
        contract_type = "Agreement"
        title_lower = title.lower()
        type_map = {
            "service": "Service Agreement", "employment": "Employment Agreement",
            "lease": "Lease Agreement", "nda": "Non-Disclosure Agreement",
            "non-disclosure": "Non-Disclosure Agreement"
        }
        for key, value in type_map.items():
            if key in title_lower:
                contract_type = value
                break

        # Effective Date Extraction
        effective_date = None
        date_patterns = [
            r'effective\s+(?:as\s+of|on)?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'dated\s+(?:as\s+of|on)?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'entered\s+into\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, self.text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                effective_date = self._normalize_date(date_str)
                if effective_date:
                    break

        return {
            "title": title,
            "contract_type": contract_type,
            "effective_date": effective_date
        }

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Converts various date string formats to YYYY-MM-DD."""
        formats_to_try = [
            '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y',
            '%Y-%m-%d', '%m/%d/%Y'
        ]
        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_str.replace(",", ""), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    def _segment_sections_and_clauses(self) -> List[SectionModel]:
        """Segments the document into sections and clauses."""
        sections = []
        current_section_content = []
        current_section_header = None

        section_header_pattern = re.compile(
            r'^(?:ARTICLE|SECTION)\s+[IVXLC\d]+[:\.]?\s+.*|^\d+\.\s+[A-Z\s]+$',
            re.IGNORECASE
        )

        for line in self.lines:
            if len(line.split()) < 10 and section_header_pattern.match(line):
                if current_section_header:
                    clauses = self._parse_clauses_from_content(current_section_content)
                    title, number = self._parse_section_header(current_section_header)
                    sections.append(SectionModel(title=title, number=number, clauses=clauses))

                current_section_header = line
                current_section_content = []
            elif current_section_header:
                current_section_content.append(line)

        if current_section_header:
            clauses = self._parse_clauses_from_content(current_section_content)
            title, number = self._parse_section_header(current_section_header)
            sections.append(SectionModel(title=title, number=number, clauses=clauses))

        return sections

    def _parse_section_header(self, header: str) -> (str, Optional[str]):
        """Extracts title and number from a section header line."""
        match = re.match(r'^(?:(ARTICLE|SECTION)\s+([IVXLC\d]+)|(\d+))\s*[:\.]?\s*(.*)', header, re.IGNORECASE)
        if match:
            groups = match.groups()
            number = groups[1] or groups[2]
            title = groups[3].strip()
            return title.title(), str(number) if number else None
        return header.title(), None

    def _parse_clauses_from_content(self, content: List[str]) -> List[ClauseModel]:
        """Parses a block of text into individual clauses."""
        clauses = []
        current_clause_text = []
        current_clause_label = ""
        clause_idx = 0

        clause_start_pattern = re.compile(r'^\s*(\d+\.\d+(?:\.\d+)*|\([a-z\d]+\))\s*([A-Z][a-z]+\.?\s)?(.*)')

        for line in content:
            match = clause_start_pattern.match(line)
            if match:
                if current_clause_text:
                    clauses.append(ClauseModel(
                        text=" ".join(current_clause_text),
                        label=current_clause_label,
                        index=clause_idx
                    ))
                    clause_idx += 1

                number_label = match.group(1)
                title_label = match.group(2)
                text_part = match.group(3).strip()

                if title_label:
                    current_clause_label = f"{number_label} {title_label.strip()}"
                else:
                    current_clause_label = number_label

                potential_title_match = re.match(r'^[A-Z][a-z]+\.?\s(.*)', text_part)
                if potential_title_match:
                     text_part = potential_title_match.group(1)

                current_clause_text = [text_part]
            else:
                if not current_clause_text and not clauses:
                    current_clause_label = ""
                current_clause_text.append(line)

        if current_clause_text:
            clauses.append(ClauseModel(
                text=" ".join(current_clause_text),
                label=current_clause_label,
                index=clause_idx
            ))

        for clause in clauses:
            clause.text = re.sub(r'\s+', ' ', clause.text).strip()

        return clauses

# =============================================================================
# 4. LAYER 5: OPTIONAL LLM ENHANCEMENT
# =============================================================================

class LLMEnhancer:
    """Uses a Generative AI to parse the contract as a fallback or for enhancement."""

    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.is_configured = True
            logging.info("Google Gemini API configured successfully.")
        except Exception as e:
            self.is_configured = False
            logging.warning(f"Failed to configure Gemini API: {e}. LLM enhancement will be disabled.")

    def enhance(self, text: str) -> Optional[ContractModel]:
        """Attempts to parse the contract text using the LLM."""
        if not self.is_configured:
            return None

        prompt = self._create_prompt(text)
        try:
            logging.info("Calling Gemini API for enhancement...")
            response = self.model.generate_content(prompt)
            cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response.text, flags=re.MULTILINE).strip()
            
            parsed_json = json.loads(cleaned_response)
            contract = ContractModel.model_validate(parsed_json)
            logging.info("LLM enhancement successful and validated.")
            return contract
        except Exception as e:
            logging.error(f"LLM enhancement failed: {e}")
            logging.error(f"LLM raw response was: {response.text if 'response' in locals() else 'N/A'}")
            return None

    def _create_prompt(self, text: str) -> str:
        """Creates the detailed prompt for the LLM."""
        return f"""
        Analyze the following contract text and convert it into a structured JSON object.
        Adhere strictly to the provided JSON schema.

        RULES:
        1.  `title`: The main title of the contract.
        2.  `contract_type`: Infer the type (e.g., "Service Agreement", "Lease Agreement").
        3.  `effective_date`: Find the effective date and format it as "YYYY-MM-DD". If not found, use `null`.
        4.  `sections`: An array of section objects. Group related clauses under a clear section header.
        5.  `sections.title`: The text of the section header (e.g., "Definitions").
        6.  `sections.number`: The number of the section (e.g., "1", "II"). If no number, use `null`.
        7.  `clauses`: An array of clause objects within a section.
        8.  `clauses.text`: The full, normalized text of the clause.
        9.  `clauses.label`: The number or letter of the clause (e.g., "1.1", "(a)"). If unlabeled, use an empty string `""`.
        10. `clauses.index`: A zero-based index, reset for each new section.
        11. Whitespace: Normalize all internal whitespace in text fields to a single space.
        12. Output MUST be a single valid JSON object, and nothing else.

        JSON SCHEMA TO FOLLOW:
        {{
          "title": "Contract Title",
          "contract_type": "Agreement Type",
          "effective_date": "YYYY-MM-DD or null",
          "sections": [
            {{
              "title": "Section Title",
              "number": "Section Number or null",
              "clauses": [
                {{
                  "text": "Clause text",
                  "label": "Label, title, and number/letter if any, otherwise empty string.",
                  "index": 0
                }}
              ]
            }}
          ]
        }}

        CONTRACT TEXT:
        ---
        {text[:15000]}
        ---
        """
# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main():
    """Main function to run the CLI tool."""
    parser = argparse.ArgumentParser(description="Parse a contract PDF into structured JSON.")
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    parser.add_argument("output_json", help="Path to write the output JSON file.")
    parser.add_argument("--force-llm", action="store_true", help="Force using LLM instead of rule-based parser.")
    args = parser.parse_args()

    # 1. Extract Text
    raw_text = get_pdf_text(args.input_pdf)
    if not raw_text:
        logging.error("Could not extract text from PDF. Exiting.")
        error_output = ContractModel(title="Error", contract_type="Error", sections=[]).model_dump_json(indent=2)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            f.write(error_output)
        sys.exit(1)
    
    final_contract = None

    # 2. Decide parsing strategy
    api_key = os.getenv("GEMINI_API_KEY")
    use_llm = args.force_llm

    if use_llm:
        if not api_key:
            logging.error("--force-llm requires the GEMINI_API_KEY environment variable to be set.")
            sys.exit(1)
        enhancer = LLMEnhancer(api_key=api_key)
        final_contract = enhancer.enhance(raw_text)
        if not final_contract:
            logging.fatal("LLM parsing was forced but failed. Exiting.")
            sys.exit(1)
    else:
        logging.info("Running rule-based parser...")
        parser = RuleBasedParser(raw_text)
        rule_based_contract = parser.parse()

        is_low_confidence = len(rule_based_contract.sections) <= 1

        if is_low_confidence and api_key:
            logging.warning("Rule-based parsing resulted in low confidence. Attempting LLM enhancement...")
            enhancer = LLMEnhancer(api_key=api_key)
            llm_contract = enhancer.enhance(raw_text)
            if llm_contract:
                final_contract = llm_contract
            else:
                logging.warning("LLM enhancement failed. Falling back to rule-based result.")
                final_contract = rule_based_contract
        else:
            if is_low_confidence:
                logging.warning("Rule-based parsing resulted in low confidence. No API key found for enhancement.")
            else:
                logging.info("Rule-based parsing successful.")
            final_contract = rule_based_contract

    # 3. Write Output
    if final_contract:
        logging.info(f"Writing output to {args.output_json}")
        with open(args.output_json, 'w', encoding='utf-8') as f:
            f.write(final_contract.model_dump_json(indent=2, by_alias=True))
        logging.info("Processing complete.")
    else:
        logging.error("Could not generate a final contract structure.")
        sys.exit(1)


if __name__ == "__main__":
    main()