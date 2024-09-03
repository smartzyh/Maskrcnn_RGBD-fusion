import torch
import cv2
import torchvision
import onnxruntime as ort
import numpy as np
import onnx

img = cv2.imread('./data/coco2022/val2022/202004010001.png', cv2.IMREAD_UNCHANGED)
print(img.shape)
# img = img[:, :, ::-1]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
img = torch.from_numpy(img.transpose((2, 0, 1)))
print(img.shape)
img = img.float().div(256)
img = np.expand_dims(img, axis=0)

model = onnx.load('./save_weights/rgbd101-real/mask.onnx')
onnx.checker.check_model(model)
output = model.graph.output
print(output)

ort_session = ort.InferenceSession('./save_weights/rgbd101-real/mask.onnx')
outputs = ort_session.run(None, {'input_1': img})
print(outputs)
