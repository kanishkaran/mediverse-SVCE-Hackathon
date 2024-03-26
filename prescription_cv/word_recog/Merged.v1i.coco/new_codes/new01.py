import os
import json
import cv2
import torch
import pandas as pd

# Function to load annotations from JSON file
def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

# Function to preprocess images (optional: convert to grayscale)
def preprocess_images(image_dir, annotations, grayscale=False):
    processed_data = []
    for image_info in annotations['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_data.append({'image_id': image_id, 'image': image})
    return processed_data

def convert_annotations_to_tensors(annotations, categories):
    annotation_tensors = []
    for annotation in annotations:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        
        # Convert bounding box coordinates to PyTorch tensor
        bbox_tensor = torch.tensor(bbox)
        
        # Find category name corresponding to category ID
        category_name = None
        for category in categories:
            if category['id'] == category_id:
                category_name = category['name']
                break
        
        # Get the index of the category name in the categories list
        class_label = categories.index(category_name)
        
        # Create annotation tensor (image_id, bbox, class_label)
        annotation_tensor = torch.tensor([image_id] + bbox + [class_label])
        annotation_tensors.append(annotation_tensor)
    
    print("Number of annotations processed:", len(annotation_tensors))
    return torch.stack(annotation_tensors)

# Define paths to dataset and annotations
dataset_dir = r'D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dset_rnamed\renamed'

# Define paths to JSON files for train, test, and valid sets
# Define paths to JSON files for train, test, and valid sets
train_json_file = 'train/updated_coco.json'
test_json_file = 'test/updated_coco.json'
valid_json_file = 'valid/updated_coco.json'

# Load annotations for train, test, and valid sets
# Verify paths to JSON files
print("Train JSON file path:", os.path.join(dataset_dir, train_json_file))
print("Test JSON file path:", os.path.join(dataset_dir, test_json_file))
print("Valid JSON file path:", os.path.join(dataset_dir, valid_json_file))

# Load annotations for train, test, and valid sets
train_annotations = load_annotations(os.path.join(dataset_dir, train_json_file))
test_annotations = load_annotations(os.path.join(dataset_dir, test_json_file))
valid_annotations = load_annotations(os.path.join(dataset_dir, valid_json_file))

# Print loaded annotations
print("Loaded train annotations:", train_annotations)
print("Loaded test annotations:", test_annotations)
print("Loaded valid annotations:", valid_annotations)



# Preprocess images for train, test, and valid sets
# train_data = preprocess_images(os.path.join(dataset_dir, 'train'), train_annotations, grayscale=True)
# test_data = preprocess_images(os.path.join(dataset_dir, 'test'), test_annotations, grayscale=True)
# valid_data = preprocess_images(os.path.join(dataset_dir, 'valid'), valid_annotations, grayscale=True)

# Extract categories from annotations
categories = train_annotations['categories']  # Assuming categories are the same across train, test, and valid sets

# Convert annotations to tensors for train, test, and valid sets
train_annotation_tensors = convert_annotations_to_tensors(train_annotations['annotations'], categories)
test_annotation_tensors = convert_annotations_to_tensors(test_annotations['annotations'], categories)
valid_annotation_tensors = convert_annotations_to_tensors(valid_annotations['annotations'], categories)

# Save annotations in CSV format (if needed)
train_df = pd.DataFrame(train_annotation_tensors.numpy(), columns=['image_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'class_label'])
test_df = pd.DataFrame(test_annotation_tensors.numpy(), columns=['image_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'class_label'])
valid_df = pd.DataFrame(valid_annotation_tensors.numpy(), columns=['image_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'class_label'])

train_df.to_csv('train_annotations.csv', index=False)
test_df.to_csv('test_annotations.csv', index=False)
valid_df.to_csv('valid_annotations.csv', index=False)
