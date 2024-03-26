import pytesseract
from PIL import Image
import cv2

# Path to your Tesseract executable (update this accordingly)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    # Open the image file
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return thresh

def detect_handwritten_text(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Convert to PIL Image
    pil_image = Image.fromarray(preprocessed_image)

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(pil_image)

    return text

# Path to your handwritten image
image_path = r'D:\code\hackathon\word_recog\doc_digit.jpg'

# Detect handwritten text
detected_text = detect_handwritten_text(image_path)
print("Detected Text:")
print(detected_text)
