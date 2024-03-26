import json
import os

def rename_images_and_update_json(json_file_path, output_dir, images_dir):
    # Load COCO JSON data
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = coco_data.get('images', [])

    for idx, image_info in enumerate(images):
        old_image_name = os.path.basename(image_info['file_name'])
        old_image_path = os.path.join(images_dir, old_image_name)
        image_extension = os.path.splitext(old_image_name)[1]
        new_image_name = f'image_{idx + 1:06d}{image_extension}'
        new_image_path = os.path.join(output_dir, new_image_name)

        try:
            # Rename image file
            os.rename(old_image_path, new_image_path)
        except Exception as e:
            print(f"Error renaming image '{old_image_path}' to '{new_image_path}': {e}")

        # Update image file name in JSON data
        image_info['file_name'] = new_image_path

    # Write updated JSON data back to file
    output_json_path = os.path.join(output_dir, 'updated_coco.json')
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f'Renamed {len(images)} images and updated JSON file: {output_json_path}')

# Example usage
json_file_path = r'word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\original\valid\_annotations.coco.json' #deleted, the files referring to the dataset downloaded from roboflow universe in json coco format
output_dir = r'word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\valid'
images_dir = r'word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\original\valid'
rename_images_and_update_json(json_file_path, output_dir, images_dir)
