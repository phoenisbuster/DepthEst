from flask import Flask, jsonify, request
import os
import base64
from celery import Celery
from time import sleep

app = Flask(__name__)

app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])
# celery.conf.update(app.config)


# Set allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Set secret key for CSRF protection
app.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@celery.task
def process_image(img_str):
    """Process uploaded image with AI model and return distance of object in frame"""
    # Decode the string back to bytes using the base64 module
    img_bytes = base64.b64decode(img_str.encode('utf-8'))
    
    # TODO: add AI model code to process image and return distance of object
    distance = 20  # Placeholder value for demonstration purposes only
    sleep(4)
    return distance


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

    # Read image bytes and process image in the background with AI model to get distance of object in frame
    img_bytes = image_file.read()
    # Encode the img_bytes argument as a string (JSON serializable object)
    img_str = base64.b64encode(img_bytes).decode('utf-8')
    task = process_image.delay(img_str)
    distance = task.get()

    # Return distance of object in frame
    return jsonify({'distance': distance}), 200


if __name__ == '__main__':
    app.run(debug=True)
