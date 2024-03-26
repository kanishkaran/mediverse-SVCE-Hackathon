import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load annotations from JSON files
def load_annotations(data_dir):
    annotations_file = os.path.join(data_dir, '_annotations.coco.json')
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)

    category_map = {category['id']: category['name'] for category in annotations_data['categories']}

    image_info = []
    for annotation in annotations_data['annotations']:
        image_id = str(annotation['image_id'])
        image_filename = next((f for f in os.listdir(data_dir) if image_id in f and f.endswith('.jpg')), None)
        if image_filename:
            image_path = os.path.join(data_dir, image_filename)
            box = annotation['bbox']
            xmin = box[0] / 640  # Assuming image width is 640
            ymin = box[1] / 640  # Assuming image height is 640
            xmax = (box[0] + box[2]) / 640
            ymax = (box[1] + box[3]) / 640
            label = category_map.get(annotation['category_id'], 'unknown')
            image_info.append((image_path, xmin, ymin, xmax, ymax, label))
        else:
            print(f"No image file found for image ID: {image_id}")

    return image_info, category_map

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))  # Resize to fit model input shape
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to create a dataset from image info
def create_dataset(image_info, batch_size=32, shuffle=False):
    image_paths = [info[0] for info in image_info]
    labels = [info[5] for info in image_info]

    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    image_ds = image_paths_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((image_ds, labels_ds))

    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_info))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Load the data for train, validation, and test sets
train_image_info, category_map = load_annotations(r'word_recog\Merged.v1i.coco\train')
val_image_info, _ = load_annotations(r'word_recog\Merged.v1i.coco\valid')
test_image_info, _ = load_annotations(r'word_recog\Merged.v1i.coco\test')

# Create datasets
train_dataset = create_dataset(train_image_info, batch_size=32, shuffle=True)
val_dataset = create_dataset(val_image_info, batch_size=32)
test_dataset = create_dataset(test_image_info, batch_size=32)

# Define the model architecture
num_classes = len(set(category_map.values()))
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)

# Save the best version of the model
model.save('best_model.h5')

# Load the saved model
loaded_model = tf.keras.models.load_model('best_model.h5')

# Make predictions on new images
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

new_image = load_and_preprocess_image(r"test_sample_images\test09.png")
predictions = loaded_model.predict(tf.expand_dims(new_image, axis=0))

# Interpret predictions
category_names = list(category_map.values())
boxes, labels, scores = [], [], []

for pred in predictions:
    for i, score in enumerate(pred):
        if score > 0.5:  # Adjust the threshold as needed
            boxes.append(pred[:4])
            labels.append(category_names[i])
            scores.append(score)

# Print predictions
for box, label, score in zip(boxes, labels, scores):
    xmin, ymin, xmax, ymax = box
    print(f"Label: {label}, Score: {score:.2f}, Box: ({xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f})")
