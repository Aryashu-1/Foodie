import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import csv

# Load model and class names
model = load_model('food_classifier.h5')
class_names = sorted(os.listdir('C:/Users/susha/code/Foodie/Indian Food Images/Indian Food Images'))  # Replace path

# Load calorie info
food_calories = {}
with open('indian_food_named_calories.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        food_name = row[0]
        calories = float(row[1])
        food_calories[food_name] = calories

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    input_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(input_array)
    pred_index = np.argmax(predictions[0])
    pred_class = class_names[pred_index]
    confidence = float(predictions[0][pred_index])
    calories = food_calories.get(pred_class, "Unknown")
    return {
        "class": pred_class,
        "confidence": round(confidence * 100, 2),
        "calories": calories
    }
