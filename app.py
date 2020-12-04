import os

from flask import Flask, render_template, request, redirect

from inference import get_prediction
from base64 import b64encode
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        image = b64encode(img_bytes).decode("utf-8")
        predict_post = get_prediction(image_bytes=img_bytes)
        return render_template('result.html', predict_post=predict_post, image=image)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
