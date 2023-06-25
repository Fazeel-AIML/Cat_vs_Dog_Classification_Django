import tensorflow as tf
import cv2

# Load the model
model = tf.keras.models.load_model('cat_vs_dog.h5')

# Load and preprocess the test image
test_img = cv2.imread('dog.jpg')
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape(1, 256, 256, 3) / 255.0

# Make predictions
predictions = model.predict(test_input)

# The predictions will be an array of probabilities
# If the value is close to 0, it indicates a cat
# If the value is close to 1, it indicates a dog
if predictions[0] < 0.5:
    print('Cat')
else:
    print('Dog')
