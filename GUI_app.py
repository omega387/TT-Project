import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import sys
import os


class NullWriter:
    def write(self, text):
        pass

    def flush(self):
        pass


if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()


def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


model = None
try:
    model_path = resource_path('cat_dog_classifier.keras')
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model:\n{e}")


def classify_image(file_path):
    if model is None:
        messagebox.showerror("Error", "Model is not loaded. Cannot predict.")
        return

    try:
        img = Image.open(file_path)
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        if prediction[0][0] > 0.5:
            result = "It's a DOG!"
            color = "green"
        else:
            result = "It's a CAT!"
            color = "blue"

        label_result.config(text=result, fg=color)
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        uploaded = Image.open(file_path)
        uploaded.thumbnail((300, 300))
        im = ImageTk.PhotoImage(uploaded)

        label_image.configure(image=im)
        label_image.image = im
        label_result.config(text="")

        classify_image(file_path)


root = tk.Tk()
root.geometry("500x600")
root.title("Cat vs Dog Classifier")
root.configure(background='#f0f0f0')

title_lbl = Label(root, text="Cat vs Dog Classifier", font=('Arial', 20, 'bold'), bg='#f0f0f0')
title_lbl.pack(pady=20)

label_image = Label(root, bg='#f0f0f0')
label_image.pack(pady=10)

label_result = Label(root, text="", font=('Arial', 18, 'bold'), bg='#f0f0f0')
label_result.pack(pady=20)

btn = Button(root, text="Select Image", command=upload_image, font=('Arial', 12), bg='#333', fg='white', padx=20,
             pady=10)
btn.pack(side="bottom", pady=30)

root.mainloop()