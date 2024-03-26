# import json
# import cv2
# import os

# def load_images_and_annotations(json_file_path, images_dir):
#     with open(json_file_path, 'r') as f:
#         coco_data = json.load(f)
        
#     images = coco_data.get('images', [])
#     annotations = coco_data.get('annotations', [])
#     categories = coco_data.get('categories', [])
    
#     category_id_to_name = {category['id']: category['name'] for category in categories}
    
#     image_data = []
#     for image_info in images:
#         image_id = image_info['id']
#         image_name = image_info['file_name']
#         image_path = os.path.join(images_dir, image_name)  # Construct absolute image path
        
#         # Load image using OpenCV
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
        
#         # Get annotations for the current image
#         image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image_id]
        
#         annotations_data = []
#         for annotation in image_annotations:
#             bbox = annotation['bbox']
#             class_id = annotation['category_id']
#             class_name = category_id_to_name.get(class_id, 'Unknown')
#             annotations_data.append({
#                 'bbox': bbox,
#                 'class_id': class_id,
#                 'class_name': class_name
#             })
        
#         image_data.append({
#             'image': image,
#             'annotations': annotations_data
#         })
    
#     return image_data

# #useage
# json_file_path = r'D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\test\updated_coco.json'
# images_dir = r'D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\test'
# image_data = load_images_and_annotations(json_file_path, images_dir)

# # Print image paths and corresponding annotations
# for idx, data in enumerate(image_data):
#     print(f"Image {idx + 1}:")
#     print("Annotations:")
#     for annotation in data['annotations']:
#         print("  Class:", annotation['class_name'])
#         print("  Bounding Box:", annotation['bbox'])
#     print()

import json
import cv2
import os

def load_images_and_annotations(json_file_path, images_dir):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    category_id_to_name = {category['id']: category['name'] for category in categories}
    
    image_data = []
    for image_info in images:
        image_id = image_info['id']
        image_name = image_info['file_name']
        image_path = os.path.join(images_dir, image_name)  # Construct absolute image path
        
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Get annotations for the current image
        image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image_id]
        
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
            'image': image,
            'annotations': annotations_data
        })
    
    return image_data

def display_image_with_annotations(image_data):
    for idx, data in enumerate(image_data):
        image = data['image'].copy()
        annotations = data['annotations']
        
        # Draw bounding boxes and annotations on the image
        for annotation in annotations:
            bbox = annotation['bbox']
            class_name = annotation['class_name']
            xmin, ymin, width, height = map(int, bbox)
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the image with annotations
        cv2.imshow(f"Image {idx + 1}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
json_file_path = r'D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\train\updated_coco.json'
images_dir = r'D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\train'
image_data = load_images_and_annotations(json_file_path, images_dir)
display_image_with_annotations(image_data)
