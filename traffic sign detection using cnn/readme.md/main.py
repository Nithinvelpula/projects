# Importing Tkinter, PIL, Numpy, pandas and model loading function from keras into the code

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import os
from pathlib import Path

# Load model without compiling to avoid legacy issues
model_path = Path("model.h5")
model = load_model(model_path, compile=False)

# Compile the model manually to ensure compatibility
model.compile(optimizer=Adam(), 
              loss=SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Load class labels
classes = pd.read_csv("labels.csv")  # If that's the correct one

# Initializing the Tkinter GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign Detection')
top.configure(background='#99cfe0')

label = Label(top, background='#99cfe0', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Function to classify the image uploaded by user
def classify(file_path):
    global label_packed
    preds = []
    image = Image.open(file_path).convert('L').resize((90, 90))
    image = np.array(image)
    preds.append(image)
    preds = np.array(preds)
    preds = preds.reshape(1, 90, 90, 1)  # Ensuring shape is (1, height, width, channels)
    prediction = model.predict(preds)
    prediction1 = np.argmax(prediction, axis=1)[0]
    sign = classes['Name'][prediction1]
    print(sign)
    label.configure(foreground='#0a0a0a', text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#e0aa99', foreground='#0a0a0a', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.6, rely=0.856)

# Function to allow the user to upload an image to classify it
def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im
    label.configure(text='')
    show_classify_button(file_path)

# Main Function
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#e0aa99', foreground='#0a0a0a', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Detect Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#99cfe0', foreground='#0a0a0a')
heading.pack()

top.mainloop()
