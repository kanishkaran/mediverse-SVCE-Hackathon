# %% [markdown]
# ## getting the data and all the necessary text files and functions

# %%
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# %matplotlib inline
import random

# %%
# # Create a folder to store data
# !mkdir sample_data/IAM

# # Download the dataset
# !wget -q https://git.io/J0fjL -O IAM_Words.zip
# !unzip -qq IAM_Words.zip -d sample_data/IAM

# # Extract the words.tgz file
# !tar -xf sample_data/IAM/IAM_Words/words.tgz -C sample_data/IAM

# # Move the words.txt file
# !mv sample_data/IAM/IAM_Words/words.txt sample_data/IAM/

# # Set the locations
# data_location = 'sample_data/IAM'
# words_txt_location = 'sample_data/IAM/words.txt'

# # Copy the splits files
# !cp -navr "/content/drive/My Drive/Colab Notebooks/OCR on IAM/train_files.txt" "/content/sample_data/IAM/"
# !cp -navr "/content/drive/My Drive/Colab Notebooks/OCR on IAM/valid_files.txt" "/content/sample_data/IAM/"
# !cp -navr "/content/drive/My Drive/Colab Notebooks/OCR on IAM/test_files.txt" "/content/sample_data/IAM/"


# %% [markdown]
# ## pre-processing

# %%
def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 3]) * 255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad


def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w < target_w and h < target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w >= target_w and h < target_h:
        new_w = target_w
        new_h = int(h * new_w / w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w < target_w and h >= target_h:
        new_h = target_h
        new_w = int(w * new_h / h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        """w>=target_w and h>=target_h """
        ratio = max(w / target_w, h / target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img


def preprocess(path, img_w, img_h):
    """ Pre-processing image for predicting """
    img = cv2.imread(path)
    img = fix_size(img, img_w, img_h)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    img /= 255
    return img

# %%
letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

num_classes = len(letters) + 1
print(num_classes)

# %% [markdown]
# ## generating all the pre-processed vectors

# %%
def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

# %%
class TextImageGenerator:

    def __init__(self, data,
                 img_w,
                 img_h,
                 batch_size,
                 i_len,
                 max_text_len):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.samples = data
        self.n = len(self.samples)
        self.i_len = i_len
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = preprocess(img_filepath, self.img_w, self.img_h)
            self.imgs[i, :, :] = img
            self.texts.append(text)

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.zeros([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * self.i_len
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i, :len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = [X_data, Y_data, input_length, label_length]
            outputs = np.zeros([self.batch_size])
            yield (inputs, outputs)

# %%
batch_size = 64
input_length = 30
max_text_len = 16
img_w = 128
img_h = 64

# %% [markdown]
# # making model

# %%
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as tf_keras_backend

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()

# %%
input_data = layers.Input(name='the_input', shape=(128,64,1), dtype='float32')  # (None, 128, 64, 1)

# Convolution layer (VGG)
iam_layers = layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(iam_layers)  # (None,64, 32, 64)

iam_layers = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)

iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(iam_layers)  # (None, 32, 8, 256)

iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv6')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)

iam_layers = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)

# CNN to RNN
iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)
iam_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)

# RNN layer
gru_1 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(iam_layers)
gru_1b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(iam_layers)
reversed_gru_1b = layers.Lambda(lambda inputTensor: tf_keras_backend.reverse(inputTensor, axes=1)) (gru_1b)

gru1_merged = layers.add([gru_1, reversed_gru_1b])
gru1_merged = layers.BatchNormalization()(gru1_merged)

gru_2 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
reversed_gru_2b= layers.Lambda(lambda inputTensor: tf_keras_backend.reverse(inputTensor, axes=1)) (gru_2b)

gru2_merged = layers.concatenate([gru_2, reversed_gru_2b])
gru2_merged = layers.BatchNormalization()(gru2_merged)

# transforms RNN output to character activations:
iam_layers = layers.Dense(80, kernel_initializer='he_normal', name='dense2')(gru2_merged)
iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

labels = layers.Input(name='the_labels', shape=[16], dtype='float32')
input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
label_length = layers.Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return tf_keras_backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


# loss function
loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([iam_outputs, labels, input_length, label_length])

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
model.summary()

# %%
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
# you will know, why there aren't any metrics

# %%
import time
from tensorflow.keras.callbacks import Callback
from datetime import datetime

class EpochTimeHistory(Callback):
    """
    a custom callback to print the time(in minutes, to console) each epoch took during.
    """
    def on_train_begin(self, logs={}):
        self.train_epoch_times = []
        self.valid_epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
            cur_epoch_time = round((time.time() - self.epoch_time_start)/60, 4)
            self.train_epoch_times.append(cur_epoch_time )
            # cur_epoch_time = datetime.strptime(str(cur_epoch_time), "%H:%M:%S.%f").strftime('%H:%M:%S')
            self.train_epoch_times.append(cur_epoch_time)
            print(" ;epoch {0} took {1} minutes.".format(epoch+1, cur_epoch_time))

    ## functions used below are for recording validation times
    def on_test_begin(self, logs={}):
        self.test_time_start = time.time()

    def on_test_end(self, logs={}):
        cur_test_time = round((time.time() - self.test_time_start)/60, 4)
        self.valid_epoch_times.append(cur_test_time)
        # cur_test_time = datetime.strptime(str(cur_test_time), "%H:%M:%S.%f").strftime('%H:%M:%S')
        print(" ;validation took {0} minutes.".format(cur_test_time))

# %%
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# model_save_cb = ModelCheckpoint(filepath='/content/drive/My Drive/Colab Notebooks/OCR on IAM/data/gru-weights-epoch{epoch:02d}-val_loss{val_loss:.3f}.h5',
#                                 verbose=1, save_best_only=False, monitor='val_loss', save_weights_only=False)
# earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
# # reduce_learning_rate_cb = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=0.00001, verbose=1)
# epoch_times = EpochTimeHistory()

# %% [markdown]
# ## plotting model statistics

# %%
import seaborn as sns

# %%
import pandas as pd

# %% [markdown]
# so, let's use the weights of 30th(labelled 29 above) and 28th(labelled 27 above) epochs.

# %% [markdown]
# ## predictions on test set and calculating WER & CER

# %%
test_image_path = r"image path goes here" #avoid using raw string use os library instead

# %% [markdown]
# make predictions using weights from various checkpoints, since that's the epoch with lowest train and validation error.

# %%
test_images_processed = []
# original_test_texts = []
# for _, (test_image_path, original_test_text) in enumerate(test_files):
temp_processed_image = preprocess(path=test_image_path, img_w=128, img_h=64)
test_images_processed.append(temp_processed_image.T)


# %%
# print(len(test_files))
print(len(test_images_processed))

# %%
test_images_processed = np.array(test_images_processed)
test_images_processed.shape

# %%
test_images_processed = test_images_processed.reshape(128, 64, 1)
test_images_processed.shape

# %%
test_images_processed = test_images_processed.reshape(128, 64, 1)
test_images_processed = test_images_processed.reshape(1, 128, 64, 1)

# %%
sns.reset_orig()
plt.figure(figsize=(3, 6))
plt.imshow(test_images_processed.reshape(128,64).T)

# %% [markdown]
# ### iam with weights of final and 30th epoch. (the one with least loss on train data)

# %%
test_images_processed.shape

# %%
iam_model_pred = Model(inputs=input_data, outputs=iam_outputs)
iam_model_pred.summary()

# %%
iam_model_pred.load_weights(filepath=r'gru-model-after-4th-session.h5') #again avoid using raw string #model path goes here #referred code from other repo for this, look in readme

# %%
test_predictions_encoded = iam_model_pred.predict(x=test_images_processed)
test_predictions_encoded.shape

# %%
# use CTC decoder to decode to text
test_predictions_decoded = tf_keras_backend.get_value(tf_keras_backend.ctc_decode(test_predictions_encoded,
                                                                                  input_length = np.ones(test_predictions_encoded.shape[0])*test_predictions_encoded.shape[1],
                                                                                  greedy=True)[0][0])
test_predictions_decoded.shape

# %%
def numbered_array_to_text(numbered_array):
    numbered_array = numbered_array[numbered_array != -1]
    return "".join(letters[i] for i in numbered_array)

# %%
# for i in range(10):
# print("original_text = ", original_test_texts[i])
print("predicted text = ", numbered_array_to_text(test_predictions_decoded))
print()

# %%



