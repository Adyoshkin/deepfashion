from flask import Flask, render_template, flash, request, redirect, url_for
from waitress import serve
from keras.models import load_model


from scripts.predict import predict_on_img

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_form():
    if request.method == 'POST':
        return predict()
    else:
        return render_template('upload.html')


@app.route('/predicted', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        predict_on_img(model, file.read(), app.config['UPLOAD_FOLDER'], filename)
        flash('Image successfully uploaded and displayed')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    model = load_model('model/best_model.hdf5')

    serve(app, host='0.0.0.0', port=8008)
