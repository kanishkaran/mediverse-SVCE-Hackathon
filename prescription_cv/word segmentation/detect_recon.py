import cv2
import numpy as np

def preprocess(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binarization and inversion
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary

def segment_words(img):
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    word_images = []
    for cnt in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Extract the region of interest (ROI)
        roi = img[y:y+h, x:x+w]
        
        # Append the ROI to the list of word images
        word_images.append((roi, (x, y, w, h)))
    
    return word_images

# Load the handwritten image
image = cv2.imread(r'test06.png') #for best practices don't use raw string, image path goes here

# Preprocess the image
preprocessed = preprocess(image)

# Segment the words
word_images = segment_words(preprocessed)

# Display each segmented word image
for i, (word_img, _) in enumerate(word_images):
    cv2.imshow(f'Word {i+1}', word_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Reconstruct the word images into a single image
reconstructed_image = np.zeros_like(preprocessed)
for word_img, bbox in word_images:
    x, y, w, h = bbox
    reconstructed_image[y:y+h, x:x+w] = word_img

# Display the reconstructed image
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
