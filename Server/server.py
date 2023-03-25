from flask import Flask, jsonify, request
import base64

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    image_file = request.files['image']
    if not image_file.filename.endswith('.png'):
        return jsonify({'error': 'File must be in PNG format.'}), 400

    with image_file.stream as image_stream:
        img_bytes = image_stream.read()
    img_base64 = base64.b64encode(img_bytes).decode('ascii')
    return jsonify({'image': img_base64}), 200


if __name__ == '__main__':
    app.run(debug=True)
