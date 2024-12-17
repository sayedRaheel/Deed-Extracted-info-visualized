import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
import textwrap
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from pdf2image import convert_from_path
from main import extract_text_from_pdf, clean_extracted_text, extract_critical_information, clean_and_convert_to_json


model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to("cuda")

def visualize_extracted_info(result, extracted_info_json, pdf_path=None, output_dir="deed_visualizations"):
    """
    Visualize the extracted information from doctr OCR results and GPT extraction
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory created/accessed at: {os.path.abspath(output_dir)}")

    # Get original PDF images
    if pdf_path is None:
        pdf_path = "./2017091400824001&page-2.pdf"  # Default path
    pdf_images = convert_from_path(pdf_path)
    
    # Parse the extracted information
    if isinstance(extracted_info_json, str):
        extracted_info = json.loads(extracted_info_json)
    else:
        extracted_info = extracted_info_json
    
    # Color mapping for different types of information
    colors = {
        'owner_name': (0, 255, 0),  # Green
        'property_address': (255, 0, 0),  # Red
        'property_parcel_id': (0, 0, 255),  # Blue
        'document_id': (255, 165, 0),  # Orange
        'legal_description': (128, 0, 128),  # Purple
        'recording_information': (255, 192, 203),  # Pink
        'grantor_name': (255, 128, 0),  # Dark Orange
        'grantee_name': (0, 255, 255),  # Cyan
        'deed_type': (128, 128, 0),  # Olive
    }
    
    # Process each page
    for page_idx, (page, pdf_image) in enumerate(zip(result.pages, pdf_images)):
        # Convert PIL Image to OpenCV format
        image = cv2.cvtColor(np.array(pdf_image), cv2.COLOR_RGB2BGR)
        
        # Get page dimensions from the image
        h, w = image.shape[:2]
        
        # Draw annotations for each block of text
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    word_text = word.value.lower()
                    
                    # Check if this word appears in any of our extracted information
                    for info_type, info_value in extracted_info.items():
                        if info_value and info_value != "Not specified in document":
                            # Convert value to string and handle lists
                            info_str = str(info_value).lower()
                            if word_text in info_str.split():
                                # Get word coordinates and scale them to image dimensions
                                coords = word.geometry
                                pts = np.array([[int(coords[0][0] * w), int(coords[0][1] * h)],
                                              [int(coords[1][0] * w), int(coords[1][1] * h)]])
                                
                                # Draw semi-transparent highlight
                                overlay = image.copy()
                                cv2.rectangle(overlay,
                                            (pts[0][0], pts[0][1]),
                                            (pts[1][0], pts[1][1]),
                                            colors.get(info_type, (100, 100, 100)),
                                            -1)
                                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
                                
                                # Draw border
                                cv2.rectangle(image,
                                            (pts[0][0], pts[0][1]),
                                            (pts[1][0], pts[1][1]),
                                            colors.get(info_type, (100, 100, 100)),
                                            1)
                                
                                # Add small label above the highlight
                                label = info_type.replace('_', ' ').title()
                                (label_w, label_h), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                cv2.putText(image,
                                          label,
                                          (pts[0][0], max(0, pts[0][1] - 5)),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4,
                                          colors.get(info_type, (100, 100, 100)),
                                          1)
        
        # Save the annotated page
        output_path = os.path.join(output_dir, f"annotated_page_{page_idx+1}.png")
        cv2.imwrite(output_path, image)
        print(f"Saved annotated page {page_idx+1} to: {output_path}")
    
    # Create summary visualization
    create_summary_visualization(extracted_info, colors, output_dir)

def create_summary_visualization(extracted_info, colors, output_dir):
    """Create a summary image with all extracted information."""
    # Create a white image
    height = 1000  # Increased height to accommodate more text
    width = 1200   # Increased width for better readability
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw title
    draw.text((20, 20), "Extracted Deed Information", fill='black', font=font)
    
    # Draw legend
    legend_y = 60
    for info_type, color in colors.items():
        # Convert BGR to RGB for PIL
        rgb_color = (color[2], color[1], color[0])
        # Draw color box
        draw.rectangle([(20, legend_y), (40, legend_y + 20)], fill=rgb_color)
        # Draw label
        draw.text((50, legend_y), info_type.replace('_', ' ').title(), 
                 fill='black', font=font)
        legend_y += 30
    
    # Draw extracted information
    y_pos = legend_y + 30
    for key, value in extracted_info.items():
        if value and value != "Not specified in document":
            # Convert value to string and handle lists
            value_str = str(value)
            
            # Draw key in corresponding color
            color = colors.get(key.lower(), (0, 0, 0))
            rgb_color = (color[2], color[1], color[0])  # Convert BGR to RGB
            draw.text((20, y_pos), f"{key.replace('_', ' ').title()}:", 
                     fill=rgb_color, font=font)
            
            # Draw value in black
            wrapped_lines = textwrap.wrap(value_str, width=60)
            for line in wrapped_lines:
                y_pos += 30
                draw.text((250, y_pos), line, fill='black', font=font)
            y_pos += 30
    
    # Save the summary
    output_path = os.path.join(output_dir, "summary.png")
    image.save(output_path)
    print(f"Saved summary visualization to: {output_path}\n")

# Example usage:
def main():
    # Input PDF path
    pdf_path = "./2017091400824001&page-2.pdf"
    
    # Process the PDF
    doc = DocumentFile.from_pdf(pdf_path)
    result = model(doc)
    cleaned_text = clean_extracted_text(result)
    extracted_info = extract_critical_information(cleaned_text)
    extracted_info_json = clean_and_convert_to_json(extracted_info)
    
    # Create output directory
    output_dir = "deed_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON to file
    json_path = os.path.join(output_dir, "extracted_info.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(extracted_info_json)
    print(f"\nSaved extracted information to: {json_path}")
    
    # Generate visualizations
    visualize_extracted_info(result, extracted_info_json, pdf_path, output_dir)

if __name__ == "__main__":
    main()