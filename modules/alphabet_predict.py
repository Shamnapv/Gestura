import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("models/model_best.h5")

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def predict_alphabet(img):

    img = img.resize((128,128))

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array,axis=0)

    predictions = model.predict(img_array)

    letter = classes[np.argmax(predictions)]

    return letter