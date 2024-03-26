import easyocr
import sqlite3


def detect_handwritten_text(image_path):
    # Create an OCR reader with the English language
    reader = easyocr.Reader(['en'])

    # Perform OCR on the image
    result = reader.readtext(image_path, detail=0)

    # Combine the detected text into a single string
    detected_text = ' '.join(result)

    return detected_text

# Path to your handwritten image
image_path = r'D:\code\hackathon\word_recog\doc_digit.jpg'

# Detect handwritten text
detected_text = detect_handwritten_text(image_path)
print("Detected Text:")
print(detected_text)


