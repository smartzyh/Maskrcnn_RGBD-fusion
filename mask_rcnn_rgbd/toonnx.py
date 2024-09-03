import numpy as np
import torch
import torchvision
import cv2
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone



# def create_model(num_classes, box_thresh=0.5):
#     backbone = resnet50_fpn_backbone()
#     model = MaskRCNN(backbone,
#                      num_classes=num_classes,
#                      rpn_score_thresh=box_thresh,
#                      box_score_thresh=box_thresh)
#
#     return model

# dummy_input = torch.randn(1, 4, 960, 1024)

rgb_path = r"E:\QT\study\DMask\pic\RGB\202004010001.png"
img_rgb = cv2.imread(rgb_path)
img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

depth_path = r"E:\QT\study\DMask\pic\Depth\202004010001.png"
img_d = cv2.imread(depth_path, -1)
img_d = np.expand_dims(img_d, axis=-1)

img = np.concatenate([img_rgb, img_d], axis=-1)
img = torch.from_numpy(img.transpose((2, 0, 1)))
img = img.float().div(256)			# 据说这里也可以用255
img = torch.unsqueeze(img, dim=0)

# model = create_model(num_classes=5)
model_path = './save_weights/onnx/coco_all/model_onnx24.pth'
model = torch.load(model_path, map_location='cpu')
# model.load_state_dict(torch.load(model_path)['model'])
model.eval()
input_names = ['image']
output_names = ['boxes', 'labels', 'scores', 'masks']

torch.onnx.export(model, img, './save_weights/onnx/coco_all/mask_coco_all.onnx', verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
