from PIL import Image

def image_to_text(image_path):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Process the image (this is a dummy processing step
            processed_img = img.convert("L")  # Convert to grayscale
            text_output = "Image processed successfully. Dummy text output."
            return text_output
    except Exception as e:
        return f"Error processing image: {e}"

# if __name__=="__main__":
#     path = input("ENterpath:")
#     print(image_to_text(path))