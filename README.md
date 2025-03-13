# 🍎 Fruit Image Classifier

This project is a **Deep Learning-based Image Classification** system that identifies different types of fruits using TensorFlow and Streamlit. The model is trained using the **VGG16 Convolutional Neural Network** architecture and hyperparameter optimization with **Optuna**.

## 🚀 Features
- **Multi-class Fruit Classification** (9 classes)
- **Streamlit Web App** for easy image upload and classification
- **Model Performance Visualization** (Accuracy and Loss graphs)
- **Interactive Feedback Buttons** for user engagement
- **Custom CSS for Modern UI**

---

## 📁 Dataset
The dataset consists of images of 9 types of fruits, organized in directories for easy loading using TensorFlow utilities.

- Apple
- Banana
- Cherry
- Chickoo
- Grapes
- Kiwi
- Mango
- Orange
- Strawberry

The images are resized to **224x224** pixels to match the input size of the **VGG16** model.

---

## 🛠️ Model Building Process

### 1. **Data Loading and Preprocessing**
- Used `image_dataset_from_directory` for efficient data loading.
- Dataset is split as follows:
  - **75%** for training
  - **16.5%** for validation
  - **8.5%** for testing
- Images were normalized by scaling pixel values.

```python
from tensorflow.keras.utils import image_dataset_from_directory

data = image_dataset_from_directory('path_to_dataset', image_size=(224,224))
```

### 2. **Model Architecture**
- Used **VGG16** as the base model (with pre-trained ImageNet weights).
- Added custom layers:
  - Additional **Conv2D** and **BatchNormalization** layers
  - **MaxPooling2D** for down-sampling
  - **Dense** layers with dropout for regularization

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')
])
```

### 3. **Hyperparameter Tuning**
- Utilized **Optuna** for efficient hyperparameter optimization.
- Tuned parameters: `learning_rate`, `dropout_rate`, `batch_size`, and `units` in Dense layers.

```python
import optuna

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    units = trial.suggest_int('units', 32, 512)
    
    model = create_model(learning_rate, dropout_rate, units)
    history = model.fit(train_data, epochs=10, validation_data=val_data, batch_size=batch_size, verbose=0)
    
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### 4. **Training the Model**
- Trained with the best hyperparameters suggested by Optuna.
- Added **EarlyStopping** and **TensorBoard** for performance monitoring.

```python
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

model = create_model(learning_rate=0.0062, dropout_rate=0.47, units=79)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tensorboard = TensorBoard(log_dir='logs')

model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[early_stopping, tensorboard])
```

---

## 📊 Model Evaluation
- Visualized **Accuracy** and **Loss** curves to monitor performance.
- Evaluated the final model on the test dataset.

```python
evaluation = model.evaluate(test_data)
```

---

## 🌐 Streamlit Web App
- Simple UI for uploading images and receiving predictions.
- Displays model statistics and allows feedback.
- Enhanced UI using custom **CSS** for better visuals.

### Main Features:
- **Upload Images** (JPEG, PNG, WEBP)
- **Instant Prediction** with visual feedback
- **Feedback Buttons** (Correct/Incorrect) to interact with the model
- **Model Statistics** displayed in the sidebar

---

## ⚙️ How to Run Locally
1. Clone the repository:
```bash
git clone https://github.com/your_username/fruit-classifier.git
cd fruit-classifier
```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📚 Technologies Used
- **Python**
- **TensorFlow & Keras**
- **Optuna** for hyperparameter tuning
- **Matplotlib & Seaborn** for data visualization
- **Streamlit** for building the web app
- **Pillow** for image processing
- **Custom CSS** for UI enhancements

---

## 🎯 Future Work
- Implement feedback-based learning to improve predictions over time.
- Extend the model to classify more fruit types.
- Deploy the app using platforms like **Streamlit Cloud** or **Heroku**.

---

## 🙌 Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvements!

---

## 📄 License
This project is open-source and available under the **MIT License**.

