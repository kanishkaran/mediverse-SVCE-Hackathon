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

# Connect to the SQLite database
conn = sqlite3.connect('medicine_database.db')
cursor = conn.cursor()

# Query the database to retrieve medicine names
cursor.execute("SELECT medicine_name FROM medicine_table")
medicine_names = [row[0] for row in cursor.fetchall()]

# Close the database connection
conn.close()

# Split the detected text into individual words
words = detected_text.split()

# List to store medicines found in the detected text
medicines_found = []

# Iterate over each word and check if it matches any medicine name
for word in words:
    for medicine_name in medicine_names:
        if medicine_name.lower() in word.lower():  # Check if the medicine name is a part of the word
            medicines_found.append(word)
            break  # Stop searching for this word once a medicine name is found

print("Medicines identified:")
print(medicines_found)
