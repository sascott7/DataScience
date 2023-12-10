import joblib
from flask import Flask, request
from numpy import argmax, asarray
from PIL import Image

model = None

def load_model():
    global model
    model = joblib.load('model.sav')

def get_prediction(img):
    predictions = model.predict(img)
    prediction_value = argmax(predictions)
    return prediction_value

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML reponse."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Flask API</h1>"
        "</body>"
        "</html>"
    )
    return body

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files.get('imagefile', '')

    image = Image.open(image)
    image = asarray(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image = image / 255.0

    prediction = get_prediction(image)
    return str(prediction)

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port='2000')