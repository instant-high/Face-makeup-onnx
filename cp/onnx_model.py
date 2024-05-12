import numpy as np
import cv2
import onnxruntime

cuda_available = onnxruntime.get_device() == 'GPU'
onnx_path = 'cp/face_parser.onnx'

providers = ["CPUExecutionProvider"]
if cuda_available:
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]

session = onnxruntime.InferenceSession(onnx_path, providers = providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
def face_parser(img):

    img = cv2.resize(img, (512, 512))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Run inference
    input_data = {input_name: img.astype(np.float32)}
    output = session.run([output_name], input_data)

    parsing = np.argmax(output[0][0], axis=0)
    
    return parsing
