
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the JSON annotations
def load_annotations(data_dir):
    annotations = os.path.join(data_dir, '_annotations.coco.json')
    with open(annotations, 'r') as f:
        annotations_data = json.load(f)

    category_map = {}
    categories = annotations_data['categories']
    for category in categories:
        category_map[category['id']] = category['name']

    image_info = {}
    for annotation in annotations_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_info:
            image_info[image_id] = {
                'boxes': [],
                'labels': [],
                'file_name': [f for f in os.listdir(data_dir) if str(image_id) in f and f.endswith('.jpg')][0]
            }
        box = annotation['bbox']
        xmin = box[0] / 640  # Assuming image width is 640
        ymin = box[1] / 640  # Assuming image height is 640
        xmax = (box[0] + box[2]) / 640
        ymax = (box[1] + box[3]) / 640
        image_info[image_id]['boxes'].append([xmin, ymin, xmax, ymax])
        image_info[image_id]['labels'].append(category_map[annotation['category_id']])

    return image_info, category_map

# Load the data
train_image_info, category_map = load_annotations(r'D:\code\hackathon\word_recog\Merged.v1i.coco\train')
val_image_info, _ = load_annotations(r'D:\code\hackathon\word_recog\Merged.v1i.coco\valid')
test_image_info, _ = load_annotations(r'D:\code\hackathon\word_recog\Merged.v1i.coco\test')

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

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dictionary(
    train_image_info,
    directory=r'D:\code\hackathon\word_recog\Merged.v1i.coco\train',
    target_size=(224, 224),
    batch_size=32,
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dictionary(
    val_image_info,
    directory=r'D:\code\hackathon\word_recog\Merged.v1i.coco\valid',
    target_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

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

new_image = load_and_preprocess_image(r"D:\code\hackathon\test_sample_images\test09.png")
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