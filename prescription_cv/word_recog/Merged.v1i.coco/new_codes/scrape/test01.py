import json

def parse_coco_json(json_file_path):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)
        
    # Extract information from the COCO JSON file
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    # Map category IDs to category names
    category_id_to_name = {category['id']: category['name'] for category in categories}
    
    # Extract image paths and annotations
    image_data = []
    for image_info in images:
        image_id = image_info['id']
        image_path = image_info['file_name']
        
        # Get annotations for the current image
        image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image_id]
        
        # Extract bounding boxes and class labels
        annotations_data = []
        for annotation in image_annotations:
            bbox = annotation['bbox']
            class_id = annotation['category_id']
            class_name = category_id_to_name.get(class_id, 'Unknown')
            annotations_data.append({
                'bbox': bbox,
                'class_id': class_id,
                'class_name': class_name
            })
        
        image_data.append({
            'image_path': image_path,
            'annotations': annotations_data
        })
    
    return image_data

# Example usage
json_file_path = r'word_recog\Merged.v1i.coco\train\_annotations.coco.json'
parsed_data = parse_coco_json(json_file_path)

# Print parsed data
for image_info in parsed_data:
    print("Image Path:", image_info['image_path'])
    print("Annotations:")
    for annotation in image_info['annotations']:
        print("  Class:", annotation['class_name'])
        print("  Bounding Box:", annotation['bbox'])
    print()
