import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_text_regions(image):
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter out small contours
            text_regions.append((x, y, w, h))
    return text_regions

def segment_words(image, text_regions):
    segmented_words = []
    for (x, y, w, h) in text_regions:
        word_image = image[y:y+h, x:x+w]
        segmented_words.append(word_image)
    return segmented_words

def main():
    # Load image
    img_path = r"Screenshot 2024-03-16 224527.png" #avoid using raw string, image path goes here
    image = cv2.imread(img_path)

    # Preprocess image
    preprocessed_image = preprocess_image(image)

    # Detect text regions
    text_regions = detect_text_regions(preprocessed_image)

    # Segment words
    segmented_words = segment_words(preprocessed_image, text_regions)

    # Display results
    for word in segmented_words:
        cv2.imshow('Segmented Word', word)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
