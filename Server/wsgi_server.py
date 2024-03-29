import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from Model import networks
from Model.transforms import Resize, NormalizeImage, PrepareForNet, CenterCrop

from flask import Flask, jsonify, request
import os

app = Flask(__name__)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_transform = Compose(
        [
            lambda img: {"image": img / 255.0},            
            Resize(
                width=384,
                height=384,
                ensure_multiple_of=32,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )
model = networks.MidasNet().to(device=device)
model.load_state_dict(torch.load("Weight/model_ckpt.pt",map_location=device))

# Set allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Set secret key for CSRF protection
app.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(img_bytes):
    """Process uploaded image with AI model and return distance of object in frame"""
    # TODO: add AI model code to process image and return distance of object
    scale = 3.159407911e-4
    shift = -0.11237868

    img_bytes= np.frombuffer(img_bytes, dtype="uint8")
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_size = img.shape[0] if img.shape[0] <= img.shape[1] else img.shape[1]
    
    img = CenterCrop(input_size)(img)
    input_batch = data_transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    pred_distance = 1 / (output*scale + shift)
    pred_distance = pred_distance[int(input_size/2)][int(input_size/2)]
    pred_distance = str(round(pred_distance,2)) #Cast2str

#     byte_im = (output/output.max()*255)
#     byte_im = cv2.imencode(".png", byte_im)[1].tobytes()

    return pred_distance#, byte_im


@app.route('/upload', methods=['POST'])
def upload():
    # Check CSRF token
    if request.headers.get('X-CSRF-Token') != app.secret_key:
        return jsonify({'error': 'Invalid CSRF token.'}), 403

    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    image_file = request.files['image']
    # Check if file has allowed extension
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'File must be in PNG, JPG, or JPEG format.'}), 400

    # Read file contents from memory, encode as base64, and return as JSON response
    img_bytes = image_file.read()
    distance = process_image(img_bytes)
    
    return jsonify({'distance': distance}), 200


if __name__ == '__main__':
    
    
    app.run(host = '0.0.0.0', port = 5000, debug =True)
