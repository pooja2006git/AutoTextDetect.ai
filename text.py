
from paddleocr import PaddleOCR
import cv2

# Set the image path
image_path = r'D:\ROHIT\Project-grader\my.jpg'

# Read the image
image = cv2.imread(image_path)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Use English language model

# Perform OCR directly on the image
result = ocr.ocr(image)

# Print OCR Results
for line in result[0]:
    print("Detected text:", line[1][0])  # Extract text from OCR result

