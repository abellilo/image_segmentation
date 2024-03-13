import keras

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

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
# model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.compile()
model_summary = model.summary()
print(model_summary)

