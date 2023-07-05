import onnxruntime as ort
import numpy as np


def save_data(data, filepath):
    data = data.reshape(-1)
    size = data.shape[0]
    with open(filepath, 'w') as f:
        for i in range(size):
            f.write('{}\n'.format(data[i]))


def run_mimo_onnx():
    onnx_path = './mimo.onnx'
    session = ort.InferenceSession(onnx_path, None)

    inputs = np.random.randn(1, 3, 224, 224)
    inputs = inputs.astype(np.float32)
    # write to text file
    outputs, features = session.run(['output', 'feature'], 
        {'x1': inputs, 'x2': inputs})
    
    save_data(inputs, 'inputs.txt')
    save_data(outputs, 'output.txt')
    save_data(features, 'feature.txt')


if __name__ == '__main__':
    run_mimo_onnx()
