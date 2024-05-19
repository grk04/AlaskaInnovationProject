import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Access the images
image1 = cv2.imread('./images/face1.jpeg')
image2 = cv2.imread('./images/face2.jpeg')

# Convert images to RGB
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Resize images to a fixed size (e.g., 128x128)
image_size = (128, 128)
image1_resized = cv2.resize(image1, image_size)
image2_resized = cv2.resize(image2, image_size)

# Normalize pixel values to range [0, 1]
image1_normalized = image1_resized / 255.0
image2_normalized = image2_resized / 255.0

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (assuming you have labels for comparison)
# train_images = np.array([image1_normalized, image2_normalized])
# train_labels = np.array([1, 0])  # Assuming image1 matches image2
# model.fit(train_images, train_labels, epochs=10)

# Perform inference to compare images
# prediction = model.predict(np.array([image1_normalized, image2_normalized]))
# print("Probability of images matching:", prediction)

# Note: Above code assumes you have labeled data for training. Adjust as per your dataset and requirements.
