import tensorflow as tf
import cv2
import json
import os

# Function to load images and annotations from JSON file
def load_images_and_annotations(json_file_path):
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
        image_path = image_name  # Assuming relative path
        
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Convert image to TensorFlow tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Normalize pixel values (optional)
        image_tensor /= 255.0  # Scale pixel values to [0, 1]
        
        # Get annotations for the current image
        image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image_id]
        
        annotations_data = []
        for annotation in image_annotations:
            bbox = annotation['bbox']
            class_id = annotation['category_id']
            class_name = category_id_to_name.get(class_id, 'Unknown')
            
            # Convert bounding box coordinates to TensorFlow tensor
            bbox_tensor = tf.convert_to_tensor(bbox, dtype=tf.float32)
            
            annotations_data.append({
                'bbox': bbox_tensor,
                'class_id': class_id,
                'class_name': class_name
            })
        
        image_data.append({
            'image': image_tensor,
            'annotations': annotations_data
        })
    
    return image_data

# Function to create an object detection model
def create_object_detection_model(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

# Define input shape and number of classes
input_shape = (640, 640, 3)  # Example input shape
num_classes = 1445  # Example number of classes

# Load dataset
base_path = r'D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed'
split = 'train'  # Choose 'train', 'test', or 'valid'
json_file_path = os.path.join(base_path, split, 'updated_coco.json')
dataset = load_images_and_annotations(json_file_path)

# Split dataset into images and annotations
images = [data['image'] for data in dataset]
annotations = [data['annotations'] for data in dataset]

# Create object detection model
model = create_object_detection_model(input_shape, num_classes)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(images, annotations, epochs=10, batch_size=32, validation_split=0.2)

# Save model
model.save("object_detection_model")
