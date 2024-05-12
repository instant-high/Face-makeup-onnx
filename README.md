# Face-makeup-onnx
Lip and hair color editor using face parsing maps.

![makeup](https://github.com/instant-high/Face-makeup-onnx/assets/77229558/b23bf1d6-910a-46cb-99ed-6a8a0ef7f7cd)

https://github.com/instant-high/Face-makeup-onnx/assets/77229558/b9bb4c3b-31c8-406b-9992-a0c99ead2424

CPU or GPU

Requirements: opencv-python, numpy, scikit-image, onnxruntime (onnruntime-gpu, cudatoolkit=11.2 cudnn=8.1.0)

.

Image: python image.py (optional --image img.jpg --result result.jpg)

Video: python video.py (optional --video video.mp4 --output output.mp4)

Webcam: python webcam.py (optional --output output.mp4)

.

Download onnx-model from https://github.com/instant-high/face-parsing-onnx/releases

To change parts and colors edit code in image.py/video.py/webcam.py

- colors = [[230, 50, 20], [20, 70, 180], [20, 70, 180]]

- parts = [table['hair'], table['upper_lip'], table['lower_lip']]

.

Original torch repository: https://github.com/zllrunning/face-makeup.PyTorch



