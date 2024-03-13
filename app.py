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

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=2)

        mask = np.maximum(mask, mask_)

    Y_train[n] = mask

#test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!!')


image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


#
# #Build UNET Model
# inputs = keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
#
# #Endoder Path (Contraction Path)
# # s = keras.layers.Lambda(lambda x: x/255)(inputs)
# c1 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
# c1 = keras.layers.Dropout(0.1)(c1)
# c1 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
# p1 = keras.layers.MaxPooling2D(2,2,)(c1)
#
# c2 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
# c2 = keras.layers.Dropout(0.1)(c2)
# c2 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
# p2 = keras.layers.MaxPooling2D(2,2,)(c2)
#
# c3 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
# c3 = keras.layers.Dropout(0.2)(c3)
# c3 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
# print(c3)
# p3 = keras.layers.MaxPooling2D(2,2,)(c3)
#
# c4 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
# c4 = keras.layers.Dropout(0.2)(c4)
# c4 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
# p4 = keras.layers.MaxPooling2D(2,2)(c4)
#
# #bottle neck
# c5 = keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
# c5 = keras.layers.Dropout(0.3)(c5)
# c5 = keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
#
# #decoder path (Expansive path)
# u6 = keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(c5)
# u6 = keras.layers.concatenate([u6, c4])
# c6 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding= 'same')(u6)
# c6 = keras.layers.Dropout(0.2)(c6)
# c6 = keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
# print(c6)
#
# u7 = keras.layers.Conv2DTranspose(64, (3,3), strides= (2,2), padding='same')(c6)
# u7 = keras.layers.concatenate([u7, c3])
# c7 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
# c7 = keras.layers.Dropout(0.2)(c7)
# c7 = keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
#
# u8 = keras.layers.Conv2DTranspose(32, (3,3), strides= (2,2), padding='same')(c7)
# u8 = keras.layers.concatenate([u8, c2])
# c8 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
# c8 = keras.layers.Dropout(0.1)(c8)
# c8 = keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c8)
# print(c8)
#
# u9 = keras.layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same')(c8)
# u9 = keras.layers.concatenate([u9, c1])
# c9 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding="same")(u9)
# c9 = keras.layers.Dropout(0.1)(c9)
# c9 = keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
# print(c9)
#
# outputs = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)
#
# model = keras.Model(inputs=[inputs], outputs = [outputs])
# model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# # model.compile()
# model_summary = model.summary()
# print(model_summary)
#
#
# ##########################################################
# #model Checkpoint
# chackpointer = keras.callbacks.ModelCheckpoint('model_for_abel.h5', verbose=1, save_best_only=True)
#
# callbacks = [
#     keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
# ]
# result = model.fit(X,Y, validation_split=0.1, epochs=100, batch_size=16, callbacks = callbacks)