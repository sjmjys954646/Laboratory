import os
import PIL
from tensorflow import keras
import numpy as np
import random
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)


# Display input image #7
display(Image(filename=input_img_paths[9]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y



def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(8, 3, strides=2, padding="same")(inputs)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(16, 3, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)

    ## stage 1

    branch_start = x

    ## stage 2

    x1_1 = x

    x1_2 = layers.Conv2D(32, 3, strides=2, padding="same")(x)

    ## stage 3

    x1_1_to_2_1 = x1_1
    x1_2_to_2_1 = layers.UpSampling2D((2, 2))(x1_2)
    x1_2_to_2_1 = layers.Conv2D(16, 3, strides=1, padding="same")(x1_2_to_2_1)

    x2_1 = layers.add([x1_1_to_2_1, x1_2_to_2_1])

    x1_1_to_2_2 = layers.Conv2D(32, 3, strides=2, padding="same")(x1_1)
    x1_2_to_2_2 = x1_2

    x2_2 = layers.add([x1_1_to_2_2, x1_2_to_2_2])

    x1_1_to_2_3_once = layers.Conv2D(32, 3, strides=2, padding="same")(x1_1)
    x1_1_to_2_3_twice = layers.Conv2D(64, 3, strides=2, padding="same")(x1_1_to_2_3_once)
    x1_2_to_2_3 = layers.Conv2D(64, 3, strides=2, padding="same")(x1_2)

    x2_3 = layers.add([x1_2_to_2_3, x1_1_to_2_3_twice])

    ## stage 4

    x2_1_to_3_1 = x2_1
    x2_2_to_3_1 = layers.UpSampling2D((2, 2))(x2_2)
    x2_2_to_3_1 = layers.Conv2D(16, 3, strides=1, padding="same")(x2_2_to_3_1)
    x2_3_to_3_1_once = layers.UpSampling2D((2, 2))(x2_3)
    x2_3_to_3_1_twice = layers.UpSampling2D((2, 2))(x2_3_to_3_1_once)
    x2_3_to_3_1_twice = layers.Conv2D(16, 3, strides=1, padding="same")(x2_3_to_3_1_twice)

    x3_1 = layers.add([x2_1_to_3_1, x2_2_to_3_1, x2_3_to_3_1_twice])

    x2_1_to_3_2 = layers.Conv2D(32, 3, strides=2, padding="same")(x2_1)
    x2_2_to_3_2 = x2_2
    x2_3_to_3_2 = layers.UpSampling2D((2, 2))(x2_3)
    x2_3_to_3_2 = layers.Conv2D(32, 3, strides=1, padding="same")(x2_3_to_3_2)

    x3_2 = layers.add([x2_1_to_3_2, x2_2_to_3_2, x2_3_to_3_2])

    x2_1_to_3_3_once = layers.Conv2D(32, 3, strides=2, padding="same")(x2_1)
    x2_1_to_3_3_twice = layers.Conv2D(64, 3, strides=2, padding="same")(x2_1_to_3_3_once)
    x2_2_to_3_3 = layers.Conv2D(64, 3, strides=2, padding="same")(x2_2)
    x2_3_to_3_3 = x2_3

    x3_3 = layers.add([x2_1_to_3_3_twice, x2_2_to_3_3, x2_3_to_3_3])

    x2_1_to_3_4_once = layers.Conv2D(32, 3, strides=2, padding="same")(x2_1)
    x2_1_to_3_4_twice = layers.Conv2D(64, 3, strides=2, padding="same")(x2_1_to_3_4_once)
    x2_1_to_3_4_third = layers.Conv2D(128, 3, strides=2, padding="same")(x2_1_to_3_4_twice)
    x2_2_to_3_4_once = layers.Conv2D(64, 3, strides=2, padding="same")(x2_2)
    x2_2_to_3_4_twice = layers.Conv2D(128, 3, strides=2, padding="same")(x2_2_to_3_4_once)
    x2_3_to_3_4 = layers.Conv2D(128, 3, strides=2, padding="same")(x2_3)

    x3_4 = layers.add([x2_1_to_3_4_third, x2_2_to_3_4_twice, x2_3_to_3_4])

    x3_4 = layers.UpSampling2D((2, 2))(x3_4)
    x3_4 = layers.UpSampling2D((2, 2))(x3_4)
    x3_3 = layers.UpSampling2D((2, 2))(x3_3)
    x3_4 = layers.UpSampling2D((2, 2))(x3_4)
    x3_3 = layers.UpSampling2D((2, 2))(x3_3)
    x3_2 = layers.UpSampling2D((2, 2))(x3_2)

    outputs = layers.concatenate([x3_1, x3_2, x3_3, x3_4])

    outputs = layers.UpSampling2D((2, 2))(outputs)
    outputs = layers.UpSampling2D((2, 2))(outputs)

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(outputs)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()


# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15

history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

# Generate predictions for all images in the validation set

val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim([0, 2.5])
plt.show()