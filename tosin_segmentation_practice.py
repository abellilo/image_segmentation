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