from django.shortcuts import render, redirect
from django.http import HttpResponse
import tensorflow as tf
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import io
import base64

# Load the model
model = tf.keras.models.load_model('MyApp\Cat_Dog_classifier.h5')

def home(request):
    classification = request.session.get('classification', 'None')
    image_url = request.session.get('image_url', '')

    # Check if no image has been uploaded
    if not image_url:
        default_image_url = '/static/default_image.png'  # Update the path to your default image
        request.session['classification'] = 'None'
        request.session['image_url'] = default_image_url

    return render(request, 'home.html', {'classification': classification, 'image_url': image_url})

def upload_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['upload']

        # Read the image data
        image_data = uploaded_file.read()

        # Decode the image data
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize the image for classification
        resized_image = cv2.resize(image, (64, 64))
        test_input = resized_image.reshape(1, 64, 64, 3) / 255.0

        # Make predictions
        predictions = model.predict(test_input)

        # Get the predicted label
        pred_label = np.argmax(predictions, axis=1)[0]

        # Define the class labels
        class_labels = ['Cat', 'Dog']

        # Get the classification
        classification = class_labels[pred_label]

        # Encode the original image to base64 format
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Store the classification and image URL in the session
        request.session['classification'] = classification
        request.session['image_url'] = encoded_image

        return redirect('MyApp:home')

    else:
        return HttpResponse("Invalid request method.")