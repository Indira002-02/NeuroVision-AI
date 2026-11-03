import os
import base64
from io import BytesIO

import flask
from flask import render_template
from torch import load, inference_mode, softmax
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Index 0 was never used in training (dataset labels are 1, 2, 3)
LABELS = ['None', 'Meningioma', 'Glioma', 'Pituitary']

INFO = {
    'Meningioma': 'Grows from the meninges, the membranes covering the brain. '
                  'Most are benign and slow-growing.',
    'Glioma': 'Arises from glial cells in the brain. Ranges from slow-growing '
              'to aggressive types.',
    'Pituitary': 'Forms in the pituitary gland at the base of the brain. '
                 'Most are benign but can affect hormones and vision.',
}

device = "cuda" if is_available() else "cpu"

resnet_model = resnet50(weights=None)
n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(Linear(n_inputs, 2048),
                             SELU(),
                             Dropout(p=0.4),
                             Linear(2048, 2048),
                             SELU(),
                             Dropout(p=0.4),
                             Linear(2048, 4))

resnet_model.to(device)
resnet_model.load_state_dict(load('./models/bt_resnet50_model.pt',
                                  map_location=DEVICE(device)))
resnet_model.eval()


def preprocess_image(image_bytes):
    transform = Compose([Resize((512, 512)), ToTensor()])
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = preprocess_image(image_bytes)
    with inference_mode():
        logits = resnet_model(tensor.to(device))
    # index 0 was never trained, so only classes 1-3 are scored
    probs = softmax(logits[0, 1:], dim=0).cpu().tolist()
    best = max(range(3), key=lambda i: probs[i])
    name = LABELS[best + 1]
    all_probs = {LABELS[i + 1]: round(p * 100, 1) for i, p in enumerate(probs)}
    return name, round(probs[best] * 100, 1), all_probs


@app.route('/', methods=['GET'])
def main():
    return flask.render_template('Diseasedet.html')


@app.route("/uimg", methods=['GET', 'POST'])
def uimg():
    if flask.request.method == 'GET':
        return flask.render_template('uimg.html')

    file = flask.request.files.get('file')
    if not file or file.filename == '':
        return flask.render_template('uimg.html')

    img_bytes = file.read()
    try:
        name, confidence, probs = get_prediction(img_bytes)
    except Exception:
        return flask.render_template('uimg.html')  # not a valid image

    mime = file.mimetype or 'image/jpeg'
    image = f"data:{mime};base64," + base64.b64encode(img_bytes).decode()
    return flask.render_template('pred.html', result=name, confidence=confidence,
                                 probs=probs, info=INFO[name], image=image)


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


if __name__ == '__main__':
    app.run(debug=True)