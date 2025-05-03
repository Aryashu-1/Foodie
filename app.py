from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

from model_utils import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    result = predict_image(filepath)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
