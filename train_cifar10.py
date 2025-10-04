import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import plot_samples, plot_history, count_predictions, CLASS_NAMES

# ---------------- Load CIFAR-10 ----------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Visualize samples
plot_samples(x_train, y_train)

# ---------------- Build CNN Model ----------------
inputs = Input(shape=(32, 32, 3))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------------- Train (5 epochs quick) ----------------
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
plot_history(history, title="Original Model")

# ---------------- Data Augmentation ----------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

aug_history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                        epochs=5, validation_data=(x_test, y_test))
plot_history(aug_history, title="With Data Augmentation")

# ---------------- Evaluation ----------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest Accuracy:", test_acc)

# ---------------- Predictions ----------------
y_pred = np.argmax(model.predict(x_test), axis=1)

# Compare ground truth vs predicted
for i in range(5):
    plt.imshow(x_test[i])
    plt.title(f"True: {CLASS_NAMES[y_test[i][0]]}, Pred: {CLASS_NAMES[y_pred[i]]}")
    plt.axis("off")
    plt.show()

# Count true vs predicted
count_predictions(y_test, y_pred)

# ---------------- Save Model ----------------
model.save("cifar10_model.h5")
print("✅ Model saved as cifar10_model.h5")
