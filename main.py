from flask import Flask, render_template, flash, request, redirect, url_for
from waitress import serve
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from base64 import b64encode

from scripts.predict import predict_on_img

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///image_storage.db"

db = SQLAlchemy(app)


class FileContents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = f'pred_{secure_filename(file.filename)}'

        pred_img = predict_on_img(model, file.read())

        new_file = FileContents(name=filename, data=pred_img.read())
        db.session.add(new_file)
        db.session.commit()

        flash('Prediction:')
        base64img = "data:image/png;base64,"+b64encode(pred_img.getvalue()).decode('ascii')
        return render_template('upload.html', content=base64img)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)


if __name__ == '__main__':
    model = load_model('model/best_model.hdf5')
    db.create_all()

    serve(app, host='0.0.0.0', port=8008)
