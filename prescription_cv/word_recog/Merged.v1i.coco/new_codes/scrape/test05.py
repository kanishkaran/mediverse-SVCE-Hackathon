import tensorflow as tf
import cv2
import json
import os
import matplotlib.pyplot as plt

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
        image_path = os.path.join(os.path.dirname(json_file_path), image_name)

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Convert image to TensorFlow tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        # Normalize pixel values
        image_tensor /= 255.0

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

def get_unique_classes_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    annotations = coco_data.get('annotations', [])
    class_ids = {annotation['category_id'] for annotation in annotations}

    return class_ids

def create_model(num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    bboxes = tf.keras.layers.Dense(4, activation='linear', name='bboxes')(x)
    class_predictions = tf.keras.layers.Dense(num_classes + 1, activation='softmax', name='classes')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[bboxes, class_predictions])

    return model


def create_dataset(image_data, batch_size=32):
    images = [data_dict['image'] for data_dict in image_data]
    image_dataset = tf.data.Dataset.from_tensor_slices(images)

    max_annotations = max(len(data_dict['annotations']) for data_dict in image_data)

    annotations = []
    for data_dict in image_data:
        annotation_tensors = [
            tf.convert_to_tensor([
                annotation['bbox'][0], annotation['bbox'][1],
                annotation['bbox'][2], annotation['bbox'][3],
                annotation['class_id']
            ], dtype=tf.float32)
            for annotation in data_dict['annotations']
        ]

        # Pad the annotation tensors to the maximum length
        padding = tf.constant([0., 0., 0., 0., 0.], shape=[1, 5], dtype=tf.float32)
        padded_annotations = tf.concat([tf.stack(annotation_tensors), padding * tf.ones([max_annotations - len(annotation_tensors), 5], dtype=tf.float32)], axis=0)

        annotations.append(padded_annotations)

    annotation_dataset = tf.data.Dataset.from_tensor_slices(annotations)

    dataset = tf.data.Dataset.zip((image_dataset, annotation_dataset))
    dataset = dataset.map(preprocess_data)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([224, 224, 3], [None, 5]))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def preprocess_data(image, annotations):
    return image, annotations

# Example usage
base_path = r'D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed'

# Load data and calculate unique classes
train_data = []
test_data = []
valid_data = []
unique_classes = set()

for split in ['train', 'test', 'valid']:
    json_file_path = os.path.join(base_path, split, 'updated_coco.json')
    image_data = load_images_and_annotations(json_file_path)
    unique_classes.update(get_unique_classes_from_json(json_file_path))

    if split == 'train':
        train_data.extend(image_data)
    elif split == 'test':
        test_data.extend(image_data)
    else:
        valid_data.extend(image_data)

num_classes = len(unique_classes)
print(f"Total number of unique classes across all splits: {num_classes}")

# Create the model
model = create_model(num_classes)

# Define loss function
def detection_loss(y_true, y_pred):
    pred_bboxes = y_pred[0]
    pred_classes = y_pred[1]

    # Calculate bounding box regression loss
    bbox_loss = tf.reduce_mean(...)  # Implement your bounding box regression loss here

    # Calculate classification loss
    class_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true[:, 4], pred_classes))

    total_loss = bbox_loss + class_loss

    return total_loss

# Compile the model
model.compile(optimizer='adam', loss=detection_loss)

# Create the training and validation datasets
train_dataset = create_dataset(train_data)
valid_dataset = create_dataset(valid_data)

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_dataset, validation_data=valid_dataset, epochs=10, callbacks=[early_stopping])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the trained model
model.save("object_detection_model")