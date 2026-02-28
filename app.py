import base64
import io
import os
# from tkinter import Image
from PIL import Image
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# GET request endpoint
@app.route('/api/data', methods=['GET'])
def get_data():
    """Handle GET requests and return JSON data"""
    return jsonify({
        'message': 'GET request successful',
        'data': [1, 2, 3, 4, 5],
        'status': 'success'
    })

import math

class layer_dense:
  def __init__(self,n_inputs,n_nuerons):
    # self.weights = np.random.randn(n_inputs,n_nuerons)
    self.weights = np.random.randn(n_inputs, n_nuerons) * np.sqrt(2.0 / n_inputs)
    self.baises = np.zeros((1,n_nuerons))
  def forward(self,inputs):
    self.inputs = inputs
    self.output = np.dot(inputs,self.weights) + self.baises
  def backward(self, dvalues):
    # davlues is the gradient from the next layer
    # shape is like (m, n_neurons)
    m = self.inputs.shape[0]

    # (n_inputs, n_neurons)
    self.dweights = np.dot(self.inputs.T, dvalues) / m
    # (1, n_neurons)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True) / m
    # (m, n_inputs)
    self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
  def forward(self, inputs):
    self.inputs= inputs
    self.output = np.maximum(0,inputs)
  def backward(self,dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0 ]=0

class Activation_softmax:
  def forward(self,inputs):
    exp_values = np.exp(inputs - np.max( inputs, axis=1, keepdims=True))
    normal_values = exp_values / np.sum(exp_values , axis=1, keepdims=True)
    self.output = normal_values


class Loss_calculation:
  def forward_loss(self, y_pred, y_actual):
    self.output = np.sum(-(y_actual)*(np.log(np.clip(y_pred,1e-7,1-1e-7))),axis=1,keepdims=True)
    data_loss = np.mean(self.output)
    return data_loss
  def backward(self,y_pred,y_true):
    samples= len(y_pred)
    self.dinputs = (y_pred-y_true)/samples


# POST request endpoint
@app.route('/api/predict', methods=['POST'])
def process_data():
    """Handle POST requests with JSON payload"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    

    # Get base64 string
    image_data = data['image']
    # Remove "data:image/png;base64," part
    image_data = image_data.split(',')[1]

    # Decode base64
    image_bytes = base64.b64decode(image_data)

    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(image)
    img_array = 255 - img_array
    # Normalize to 0-1 range for neural network
    img_array = img_array / 255.0

    # Flatten to 1D array (28x28=784) for dense layer input
    img_flat = img_array.flatten().reshape(1, -1)
    # print(782)
    # print(img_flat)
    print(','.join(map(str, img_flat.flatten())))
    # print("Shape:", img_array.shape)  # (28, 28)
    # print("Pixel values:\n", img_array)

    layer1 = layer_dense(784,256)
    layer2 = layer_dense(256,62)
    # layer3 = layer_dense(128,62)


    activation1 = Activation_ReLU()
    # activation2 = Activation_ReLU()
    activation2 = Activation_softmax()

    # allloss = Loss_calculation()

    if(os.path.exists("model.npz")):
        model = np.load("model.npz")
        layer1.weights = model["layer1_w"]
        layer1.baises = model["layer1_b"]
        layer2.weights = model["layer2_w"]
        layer2.baises = model["layer2_b"]

        # Forward pass
    layer1.forward(img_flat)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    prediction = np.argmax(activation2.output,axis=1)

    print("Predicted class:", prediction[0])
    # print("True class:", y_train[1036])

    mapping={}
    with open("emnist-byclass-mapping.txt") as file:
        for line in file:
            label , ascie = line.strip().split()
            mapping[int(label)]= chr(int(ascie))
    print(mapping[prediction[0]])
    
    # Process the received data
    result = {
        'message': 'POST request received and processed',
        'received_data': mapping[prediction[0]],
        'status': 'success'
    }
    return jsonify(result), 200

# GET and POST on same endpoint
@app.route('/api/info', methods=['GET', 'POST'])
def handle_info():
    """Handle both GET and POST requests"""
    if request.method == 'GET':
        return jsonify({'method': 'GET', 'info': 'This is a GET request'})
    elif request.method == 'POST':
        data = request.get_json()
        return jsonify({'method': 'POST', 'received': data})

# Home route
@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({'message': 'Welcome to Flask API'})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)