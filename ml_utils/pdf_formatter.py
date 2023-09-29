import numpy as np
import os
import PyPDF2
import shutil
import pathlib
from pdf2image import convert_from_path
import pytesseract
import cv2
import shutil
from .directory_maker import subdir_maker

pytesseract.pytesseract.tesseract_cmd =r"C:\Program Files\Tesseract-OCR\tesseract"

def pdf_reader(file):
    read=convert_from_path(file, dpi=400)
    for i in read:
        image=cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
        # Grayscale, Gaussian blur, Otsu's threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening
        text=pytesseract.image_to_string(invert,lang='eng', config='--psm 6')
    text=text.replace('\n',' ')
    return text

def split_pdf_template(input_path, final_output_path):
    file_list = os.listdir(input_path)
    doc_list=[]
    # Step 3: Iterate over each file in the list
    for file_name in file_list:
        
        file_name=file_name.replace(' ','_')
        # Construct the full file path
        file_path = os.path.join(input_path, file_name)
        
# ********************** UPDATE FUNCTION IF FILE ALREADY EXISTS!!! ************************************** #    

        new_folder, name = subdir_maker(file_path, final_output_path)

        with open(file_path, 'rb') as file:
            read_pdf = PyPDF2.PdfReader(file)

            for page_number, page in enumerate(read_pdf.pages, start=1):
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(page)

                output_path = f"{name}_{page_number}.pdf"
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
                shutil.move(name+"_"+str(page_number)+".pdf", new_folder)
                doc_list += [(new_folder,output_path, pdf_reader(pathlib.Path(new_folder,output_path)))]
                
    return doc_list
