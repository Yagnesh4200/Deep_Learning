
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load ResNet50 model
model = ResNet50(weights='imagenet')

# Define a function to preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Define a function to predict and display the category of an image
def predict_and_display(img_path):
    img_array = preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the image using matplotlib
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')

    # Annotate the image with predictions
    annotations = "\n".join([f"{i + 1}: {label} ({score:.2f})" for i, (_, label, score) in enumerate(decoded_predictions)])
    plt.annotate(annotations, xy=(0, 1), xytext=(10, -10), va='top', ha='left', color='white', fontsize=10,
                 bbox=dict(boxstyle='round', alpha=0.5, facecolor='gray'))

    # Display the image within the notebook output
    display(IPImage(filename=img_path, width=200, height=200))

    plt.show()

# Example usage with a directory containing images
#Enter the directory of the plane images
dataset_directory = r'/content/drive/MyDrive/usecasephotos'
classify_images_in_directory(dataset_directory)