import streamlit as st
import numpy as np
import time
import tensorflow as tf
from PIL import Image
from enum import Enum

# -------------------- CSS for Styling -------------------- #
custom_css = """
<style>
    body {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stImage img {
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
    }
    .stProgress>div>div>div>div {
        background-color: #1DB954;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------- Enum for Fruit Labels -------------------- #
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

# -------------------- Model Loader -------------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Models/FruitClassifier.h5')

# -------------------- Image Classifier -------------------- #
def classify_image(image, model):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return np.argmax(predictions)

# -------------------- Display Function -------------------- #
def display_image(image, prediction):
    with st.container(border=True):
        st.image(image, caption=f'Predicted Fruit: {Fruit(prediction).name}', use_container_width=True)

# -------------------- Streamlit UI -------------------- #
st.title('🍎 Image Fruit Classifier')

# Sidebar for Model Statistics
side_bar = st.sidebar
side_bar.title('📊 Model Statistics')
side_bar.write('**_Accuracy:_**')
side_bar.image('Plots/Accuracy.png')
side_bar.write('**_Loss:_**')
side_bar.image('Plots/Loss.png')

# File Uploader
upload = st.file_uploader('📂 Upload The Fruit Image Here...', type=['jpeg', 'webp', 'png', 'jpg'])

if upload is not None:
    progress = st.progress(0, text='🔄 Processing Image...')
    model = load_model()

    for percent in range(0, 101, 20):
        time.sleep(0.05)
        progress.progress(percent)

    prediction = classify_image(upload, model)
    display_image(upload, prediction)

    # Feedback Button
    if st.button('✅ Correct Prediction'):
        st.success('Glad to hear that! 🎉')
        st.ballons()
    elif st.button('❌ Incorrect Prediction'):
        st.warning('Sorry about that! Model will learn better next time.')

else:
    st.info('📤 Upload an image to classify it.')
