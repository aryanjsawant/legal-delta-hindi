import PyPDF2
import os

def extract_pdf_to_text(pdf_path, txt_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += f"\\n--- Page {page_num + 1} ---\\n"
                text += page.extract_text() + "\\n"
            
        with open(txt_path, 'w', encoding='utf-8') as out_file:
            out_file.write(text)
        print(f"Successfully extracted {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

extract_pdf_to_text("LegalΔ- Enhancing Legal Reasoning in LLMs via Reinforcement Learning with Chain-of-Thought Guided Information Gain.pdf", "legal_delta.txt")
extract_pdf_to_text("nyayaanumana.pdf", "nyayaanumana.txt")
