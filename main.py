        # Process the uploaded PDF file
import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
from doctr.models import ocr_predictor
import openai
import re
#from openai import OpenAI
import os
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to("mps")

# import the OpenAI Python library for calling the OpenAI API
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
#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

# Function to extract text from PDF using OCR
def extract_text_from_pdf(pdf_file):
    doc = DocumentFile.from_pdf(pdf_file)
    result = model(doc)  # Assuming 'model' is your OCR predictor
    return result

# Function to clean extracted text
def clean_extracted_text(result):
    extracted_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = ' '.join(word.value for word in line.words)
                extracted_text.append(line_text)
    return '\n'.join(extracted_text)

# Function to extract critical information using a language model

def extract_critical_information(text):
    # Here you would integrate your OpenAI language model (LLM) or similar
    # For demonstration, let's assume it returns a mock JSON
    #print(text)
    MODEL = "gpt-4o"
    #response = client.chat.completions.create(
    response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": """You are a property deed expert attorney for the US with the following responsibilities:
            1. Extract only explicitly stated information from property deeds
            2. Mark missing information as "Not specified in document"
            3. Maintain exact legal language for crucial elements
            4. Flag any ambiguities or inconsistencies
            5. Never infer or assume information not present in the document"""},
        {"role": "user", "content": f"Given the raw information extracted using OCR from a PDF, you should extract the most important parts of the deed such as the owner's name, property parcel id, address, and any other important factors. The OCR results are stored in the variable `result`.\n\nOCR Result:\n{text}\n\nPlease provide the extracted information in JSON format.\n\nExample JSON output:\n{{\n  \"owner_name\": \"\",\n  \"property_address\": \"\",\n  \"property_parcel_id\": \"\",\n  \"document_id\": \"\",\n  \"legal_description\": \"\",\n  \"grantor_name\": \"\",\n  \"grantee_name\": \"\",\n  \"deed_type\": \"\",\n  \"liens_and_encumbrances\": \"\",\n  \"signatures\": \"\",\n  \"notarization_details\": \"\",\n  \"recording_information\": \"\",\n  \"consideration\": \"\",\n  \"habendum_clause\": \"\",\n  \"warranty_clauses\": \"\",\n  \"tax_information\": \"\",\n  \"title_insurance_details\": \"\"\n}}Extracted Information JSON:, Warning: Do not make up fake information"}],
    temperature=0,
    )
# 
    extracted_info= (response.choices[0].message.content)
    return extracted_info

def clean_and_convert_to_json(input_string):
    # Remove the markdown code block indicators and any leading/trailing whitespace
    cleaned_string = input_string.strip()
    cleaned_string = re.sub(r'^```json\s*|\s*```$', '', cleaned_string, flags=re.MULTILINE)
    
    # Remove any non-printable characters except newlines
    cleaned_string = ''.join(char for char in cleaned_string if char.isprintable() or char in '\n\r')
    
    # Ensure the string starts with { and ends with }
    cleaned_string = cleaned_string.strip()
    if not cleaned_string.startswith('{'):
        cleaned_string = '{' + cleaned_string
    if not cleaned_string.endswith('}'):
        cleaned_string = cleaned_string + '}'
    
    try:
        # Parse the string as JSON
        data = json.loads(cleaned_string)
        # Convert back to a JSON string with proper formatting
        cleaned_json = json.dumps(data, indent=2)
        return cleaned_json
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic string:\n{cleaned_string}")
        return None  

result = extract_text_from_pdf("./2017091400824001&page-2.pdf")
cleaned_text = clean_extracted_text(result)
extracted_info = extract_critical_information(cleaned_text)
extracted_info = clean_and_convert_to_json(extracted_info)
extracted_info_json = json.loads(extracted_info)
extracted_info_json