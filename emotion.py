import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model(r"C:\Users\Abdullah\Downloads\emotion_model.keras")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy',
               'Neutral', 'Sad', 'Surprise']

st.title("Facial Emotion Recognition")
st.write("Upload a face image or take a photo to predict the emotion.")

option = st.radio("Choose an option:", ["Upload Image", "Camera"])

if option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
else:
    image_file = st.camera_input("Take a photo")

if image_file is not None:

    image = Image.open(image_file)
    st.image(image, caption="Your Image", width=300)

    image = image.convert('L')
    image = image.resize((48, 48))

    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 48, 48, 1)

    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction)

    st.success(f"Predicted Emotion: {class_names[predicted_class]}")

    st.write("Emotion Probabilities:")

    for i in range(7):
        st.write(f"{class_names[i]}: {prediction[0][i] * 100:.2f}%")
