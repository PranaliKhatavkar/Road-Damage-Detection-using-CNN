# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:49:33 2026

@author: Pranali
"""
import tensorflow as tf
print(tf.__version__)
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
print("Pillow is working")
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "road_damage_cnn_model.keras"
)
# =========================
# Class names
# =========================

class_names = [
    "Crack",
    "Normal",
    "Pothole"
]
# =========================
# Prediction function
# =========================

def predict_image():

    file_path = filedialog.askopenfilename(
        title="Select Road Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if file_path == "":
        return

    # Load image
    image = Image.open(file_path).convert("RGB")

    # Display original image
    display_image = image.copy()
    display_image.thumbnail((400, 300))

    photo = ImageTk.PhotoImage(display_image)

    image_label.config(image=photo)
    image_label.image = photo

    # Resize for CNN
    image = image.resize((224, 224))

    # Convert image to NumPy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        np.max(prediction) * 100
    )

    # Display result
    result_label.config(
        text=f"Prediction: {predicted_class}\n"
             f"Confidence: {confidence:.2f}%"
    )


# =========================
# Create GUI
# =========================

root = tk.Tk()

root.title("Road Damage Detection Using CNN")

root.geometry("600x600")


# Heading
title_label = tk.Label(
    root,
    text="Road Damage Detection",
    font=("Arial", 22, "bold")
)

title_label.pack(pady=20)


# Upload button
upload_button = tk.Button(
    root,
    text="Upload Road Image",
    font=("Arial", 14),
    command=predict_image
)

upload_button.pack(pady=20)


# Image display
image_label = tk.Label(root)

image_label.pack(pady=20)


# Result
result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=20)


# Start application
root.mainloop()