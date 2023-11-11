from flask import Flask, render_template, request, send_from_directory
from predictor import predict_image
import os
from PIL import Image
from io import BytesIO
import requests

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            path="./uploads/{}".format(file.filename)
            prediction = predict_image(path)
            print(f"[SERVER] The prediction is: {prediction}")
            return render_template('index.html', errors = prediction, image = file.filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict_url', methods=["POST"])
def predict_from_url():
    url = request.form["url"]
    response = requests.get(url)
    image_stream = BytesIO(response.content)
    image = Image.open(image_stream)
    filename = url.split('/')[-1]
    if not filename:
        import uuid
        filename = str(uuid.uuid4()) + '.jpg'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(file_path)
    prediction = predict_image(file_path)
    return render_template('index.html', errors=prediction, image=filename)


if __name__=='__main__':
    folder_path = r"D:\Project_FaceNet\uploads"
    file_list = os.listdir(folder_path)
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    app.run(debug=True)