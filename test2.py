from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the model
model = keras.models.load_model('modelo2.h5')

img = Image.open('./6.png')

# Convert the image to grayscale
img_gray = img.convert('L')

# Resize the grayscale image to match the input shape of the model
img_resized = img_gray.resize((28, 28))

# Convert the resized image to a numpy array
np_frame_resized = np.array(img_resized)

# Normalize the pixel values
image = np_frame_resized / 255.0

# Reshape the image to match the model's input shape
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make the prediction
predictions = model.predict(image)
# print('>>>',predictions)
prediction = np.argmax(predictions[0])
print('Prediction:', prediction)
