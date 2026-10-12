import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "val"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fruit_detection_model.keras"
)


# Create models folder
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 16

EPOCHS = 10


# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# --------------------------------------------------
# DISPLAY CLASS NAMES
# --------------------------------------------------

print("\nFruit classes:")

for class_name, class_index in train_data.class_indices.items():
    print(class_index, "=", class_name)


# --------------------------------------------------
# CREATE CNN MODEL
# --------------------------------------------------

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        5,
        activation="softmax"
    )
])


# --------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# SHOW MODEL SUMMARY
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nStarting fruit detection model training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model.save(MODEL_PATH)

print("\n----------------------------------------")
print("Training completed successfully!")
print("----------------------------------------")

print("\nModel saved at:")

print(MODEL_PATH)

print("\nFinal training accuracy:")

print(
    round(
        history.history["accuracy"][-1] * 100,
        2
    ),
    "%"
)

print("\nFinal validation accuracy:")

print(
    round(
        history.history["val_accuracy"][-1] * 100,
        2
    ),
    "%"
)