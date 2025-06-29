# SmartBridge_ai_ml
HematoVision: AI-Powered Blood Cell Classification

----------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === 1. Set Paths ===
train_dir = 'C:\project\TRAIN'
val_dir = 'C:\project\TEST'
img_size = (224, 224)
batch_size = 32
num_classes = 4

# === 2. Data Augmentation & Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# === 3. Load Pre-trained Model (without top layers) ===
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# === 4. Custom Classification Head ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === 5. Compile Model ===
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === 6. Train the Model ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# === 7. Fine-Tune: Unfreeze top layers of base model ===
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# === 8. Plot Accuracy ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()

# === 9. Save Model ===
model.save("hematovision_model.h5")

--------------------------------------------------------------------------------------------------------------------------------------------------------------

**#predict.py**
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("hematovision_model.h5")

# Class names as used during training
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence = round(np.max(prediction) * 100, 2)
    return class_label, confidence


**#app.py**
from flask import Flask, render_template, request
from predict import predict_image
import os

app = Flask(_name_)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            return render_template('index.html', label=label, confidence=confidence, image=file.filename)
    return render_template('index.html', label=None)

if _name_ == '_main_':
    app.run(debug=True)

-------------------------------------------------------------------------------------------------------------------------------------

<!DOCTYPE html>
<html>
<head>
    <title>HematoVision - Predict Blood Cell</title>
</head>
<body style="text-align:center; font-family:sans-serif;">
    <h2>Upload Blood Cell Image</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Predict</button>
    </form>

    {% if label %}
        <h3>Prediction: {{ label }}</h3>
        <p>Confidence: {{ confidence }}%</p>
        <img src="{{ url_for('static', filename='../uploads/' + image) }}" width="300">
    {% endif %}
</body>
</html>

-----------------------------------------------------------------------------------------------------------------------------------
