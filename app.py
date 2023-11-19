import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2


# load the trained model
model = tf.keras.models.load_model("trained_model_vgg.h5")

# define labels for binary classification
class_labels = ['Cat', 'Dog']


# streamlit app
st.title("Dog and Cat Classification Project")
st.markdown("By pxxthik")
st.write("Upload an image to predict whether it's a Dog or a Cat!")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if st.button("Predict"):

    if uploaded_file is not None:
        # convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), np.uint8)
        test_img = cv2.imdecode(file_bytes, 1)

        # Preprocess the image
        test_img = cv2.resize(test_img, (150, 150))
        test_input = test_img.reshape((1, 150, 150, 3)) / 255.0

        # make a prediction using the model
        prediction = model.predict(test_input)
        predicted_class = class_labels[int(prediction[0][0] > 0.5)]

        # display uploaded image and prediction
        # st.image(test_img, caption=f"Uploaded Image: {predicted_class}", use_column_width=True)
        st.header(f"Prediction: {predicted_class}")

        # display the image using matplotlib
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {predicted_class}")
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
