"""
inference using onnx engine
"""
import cv2
import numpy as np
import onnxruntime as ort


class OnnxDemo(object):
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, None)
    
    def infer(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        # resize
        image = cv2.resize(image, (112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = (image - 0.5) / 0.5
        image = image.astype(np.float32)

        outputs = self.sess.run(['output'], {'input': image})
        return outputs[0]

    def get_input_info(self):
        info = []
        for item in self.sess.get_inputs():
            name = item.name
            shape = item.shape
            info.append({
                'name': name, 
                'shape': shape
            })
        return info
    
    def get_output_info(self):
        info = []
        for item in self.sess.get_outputs():
            name = item.name
            shape = item.shape
            info.append({
                'name': name, 
                'shape': shape
            })
        return info


if __name__ == '__main__':
    demo = OnnxDemo('./arcface_resnet50.onnx')
    image_path = './sample/sample1.jpg'
    out = demo.infer(image_path)
    # print(out.shape)
    print(out[0][:10])
