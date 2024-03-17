import keras
import os
import numpy as np
import random

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_IMAGE_PATH = 'train/image/'
TRAIN_MASK_PATH = 'train/mask/'
TEST_IMAGE_PATH = 'test/image/'

train_data = next(os.walk(TRAIN_IMAGE_PATH))[2]
train_mask_data = next(os.walk(TRAIN_MASK_PATH))[2]
test_data = next(os.walk(TEST_IMAGE_PATH))[2]

X_train = np.zeros((len(train_data), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_mask_data), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_test = np.zeros((len(test_data), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_data), total=len(train_data)):
    path = TRAIN_IMAGE_PATH
    # print(n)
    # print(path + id_)
    img = imread(path + id_)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

for n, id_ in tqdm(enumerate(train_mask_data), total=len(train_mask_data)):
    mask_path = TRAIN_MASK_PATH
    mask = imread(mask_path + id_)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    Y_train[n] = mask

for n, id_ in tqdm(enumerate(test_data), total=len(test_data)):
    test_path = TEST_IMAGE_PATH
    test = imread(test_path + id_)
    test = resize(test, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = test

print('Done!!')
print(Y_train.dtype)


image_x = random.randint(0, len(train_data))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


#Build UNET Model
inputs = keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

#Endoder Path (Contraction Path)
# s = keras.layers.Lambda(lambda x: x/255)(inputs)
c1 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = keras.layers.Dropout(0.1)(c1)
c1 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = keras.layers.MaxPooling2D(2,2,)(c1)

c2 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = keras.layers.Dropout(0.1)(c2)
c2 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = keras.layers.MaxPooling2D(2,2,)(c2)

c3 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = keras.layers.Dropout(0.2)(c3)
c3 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
print(c3)
p3 = keras.layers.MaxPooling2D(2,2,)(c3)

c4 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = keras.layers.Dropout(0.2)(c4)
c4 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = keras.layers.MaxPooling2D(2,2)(c4)

#bottle neck
c5 = keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = keras.layers.Dropout(0.3)(c5)
c5 = keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#decoder path (Expansive path)
u6 = keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(c5)
u6 = keras.layers.concatenate([u6, c4])
c6 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding= 'same')(u6)
c6 = keras.layers.Dropout(0.2)(c6)
c6 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
print(c6)

u7 = keras.layers.Conv2DTranspose(64, (3,3), strides= (2,2), padding='same')(c6)
u7 = keras.layers.concatenate([u7, c3])
c7 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = keras.layers.Dropout(0.2)(c7)
c7 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = keras.layers.Conv2DTranspose(32, (3,3), strides= (2,2), padding='same')(c7)
u8 = keras.layers.concatenate([u8, c2])
c8 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = keras.layers.Dropout(0.1)(c8)
c8 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c8)
print(c8)

u9 = keras.layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same')(c8)
u9 = keras.layers.concatenate([u9, c1])
c9 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding="same")(u9)
c9 = keras.layers.Dropout(0.1)(c9)
c9 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
print(c9)

outputs = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = keras.Model(inputs=[inputs], outputs = [outputs])
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# model.compile()
model_summary = model.summary()
print(model_summary)
