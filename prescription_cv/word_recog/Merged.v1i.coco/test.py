import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the JSON annotations and images
def load_data(data_dir):
    images = []
    boxes = []
    labels = []
    category_map = {}

    annotations = os.path.join(data_dir, '_annotations.coco.json')
    with open(annotations, 'r') as f:
        annotations_data = json.load(f)

    categories = annotations_data['categories']
    for category in categories:
        category_map[category['id']] = category['name']

    image_files = os.listdir(data_dir)

    for annotation in annotations_data['annotations']:
        image_id = annotation['image_id']
        image_filename = [f for f in image_files if str(image_id) in f and f.endswith('.jpg')][0]
        image_path = os.path.join(data_dir, image_filename)
        image = load_img(image_path)
        image_array = img_to_array(image)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        images.append(image_array)

        box = annotation['bbox']
        xmin = box[0] / image.size[0]
        ymin = box[1] / image.size[1]
        xmax = (box[0] + box[2]) / image.size[0]
        ymax = (box[1] + box[3]) / image.size[1]
        boxes.append([xmin, ymin, xmax, ymax])

        labels.append(category_map[annotation['category_id']])

    return np.array(images), np.array(boxes, dtype=np.float32), np.array(labels)

# Load the data
train_images, train_boxes, train_labels = load_data(r'word_recog\Merged.v1i.coco\train')
val_images, val_boxes, val_labels = load_data(r'word_recog\Merged.v1i.coco\valid')
test_images, test_boxes, test_labels = load_data(r'word_recog\Merged.v1i.coco\test')

# Define the model
model = keras.models.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(set(train_labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, [train_boxes, train_labels], epochs=10, validation_data=(val_images, [val_boxes, val_labels]))

# Save the best version of the model
model.save('best_model.h5')

# Load the saved model
loaded_model = keras.models.load_model('best_model.h5')

# Make predictions on new images
def load_and_preprocess_image(image_path):
    image = load_img(image_path)
    image_array = img_to_array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

new_image = load_and_preprocess_image(r"test_sample_images\test09.png")
predictions = loaded_model.predict(np.expand_dims(new_image, axis=0))

# Interpret predictions
category_names = list(category_map.values())
boxes, labels, scores = [], [], []

for box, label, score in zip(predictions[0], predictions[1], predictions[2]):
    if score > 0.5:  # Adjust the threshold as needed
        boxes.append(box)
        labels.append(category_names[label])
        scores.append(score)

# Print predictions
for box, label, score in zip(boxes, labels, scores):
    xmin, ymin, xmax, ymax = box
    print(f"Label: {label}, Score: {score:.2f}, Box: ({xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f})")
    

    
