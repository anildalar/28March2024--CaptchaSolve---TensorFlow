import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# (x, y), (a, b) = ((7,7),(2,2),(),(),())
# (x, y), (a, b) = someFunction()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy)

# Save the model using TensorFlow's built-in method
model.save('modelo.h5')


# Load the model
loaded_model = keras.models.load_model('modelo.h5')

predictions = loaded_model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)  # Extracting predicted labels


n = 10 # Number of images to visualize
plt.figure(figsize=(10, 10))
for i in range(n):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {predicted_labels[i]}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()


    

