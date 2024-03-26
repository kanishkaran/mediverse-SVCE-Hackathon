import easyocr
import cv2

def detect_and_display_text(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create an OCR reader with the English language
    reader = easyocr.Reader(['en'])

    # Perform OCR on the image
    result = reader.readtext(image_path)

    # Draw bounding boxes around the detected text
    for detection in result:
        # Extract coordinates
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        text = detection[1]

        # Draw the bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Put text above the bounding box
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image with bounding boxes
    cv2.imshow('Detected Text', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your handwritten image
image_path = r'image path goes here'

# Detect and display handwritten text with bounding boxes
detect_and_display_text(image_path)
