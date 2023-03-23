from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    
    image = request.files['image']
    if not image.filename.endswith('.png'):
        return jsonify({'error': 'File must be in PNG format.'}), 400
    
    img_base64 = base64.b64encode(image.read()).decode('ascii')
    return jsonify({'image': img_base64}), 200




if __name__ == '__main__':
    app.run(debug=True)