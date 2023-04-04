from flask import Flask, jsonify, request
import os

app = Flask(__name__)

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
    distance = 10  # Placeholder value for demonstration purposes only
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

    # Read file contents from memory, encode as base64, and return as JSON response
    img_bytes = image_file.read()
    distance = process_image(img_bytes)
    return jsonify({'distance': distance}), 200


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug =True)
