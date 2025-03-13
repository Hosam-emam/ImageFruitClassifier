import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import  tensorflow
from PIL import Image
from enum import Enum

class Fruit(Enum):
    Apple = 0
    Banana = 1
    Cherry = 2
    Chickoo = 3
    Grapes = 4
    Kiwi = 5
    Mango = 6
    Orange = 7
    Strawberry = 8

@st.cache_resource
def load_model():
    model = tensorflow.keras.models.load_model(r'Models/FruitClassifier.h5')
    return model

def classify_image(image_path,model):
    img = Image.open(image_path)
    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicition = np.argmax(predictions)
    return predicition


def display_image(image_path,prediction):
    img = Image.open(image_path)
    with st.container(border=True):
        if prediction is not None:
            st.image(img)
            st.caption(f'The Predicted Fruit: {Fruit(prediction).name}')
        else:
            st.image(img)
            st.caption('There Is No Fruit Detected')


st.title('Image Fruit Classifier')

side_bar = st.sidebar
side_bar.title(r'Model\'s Statistics')
side_bar.write('Accuracy ⬇️')
side_bar.image(r'Plots/Accuracy.png')
side_bar.divider()
side_bar.write('Loss ⬇️')
side_bar.image(r'Plots/Loss.png')

upload = st.empty()

with st.container(border=True):
    upload = st.file_uploader('Upload The Fruit Image Here...', type=['jpeg','webp','png'])

if upload is not None:

    progress = st.progress(0,text='Models Progress')
    for i in range(0,101, 10):
        time.sleep(0.01)
        progress.progress(i)
    model = load_model()
    prediction = classify_image(upload,model)
    display_image(upload, prediction=prediction)

    st.write('Press the button if the prediction is correct')
    button = st.button('Correct')
    if button:
        st.balloons()

