import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():

    # root = r'E:\Data\RGBD\nflg-height'
    # root_d = root + '\\' + 'concat'
    # root_rgb = root + '\\' + 'Image'

    root = r'E:\QT\study\DMask\pic'
    root_d = root + '\\' + 'concat'
    root_rgb = root + '\\' + 'RGB'

    d_names = os.listdir(root_d)
    rgb_names = os.listdir(root_rgb)

    num_classes = 4  # 不包含背景
    box_thresh = 0.8
    weights_path = "./save_weights/onnx/softnms/model_45.pth"
    # img_path = "./test/111.png"
    label_json_path = './coco91_indices.json'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights      weights_path保存训练模型的路径
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    for d_name in d_names:
        img_path = os.path.join(root_d, d_name)
        # load image
        assert os.path.exists(img_path), f"{img_path} does not exits."
        original_img = Image.open(img_path).convert('RGBA')
        # img1 = np.array(original_img)
        # print(img1.shape)
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        # data_transform = transforms.Compose([transforms.ToTensor(),
        #                                     transforms.RandomHorizontalFlip(0.5)])
        img = data_transform(original_img)
        # print(img.size())
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        # print(img.size())
        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 4, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                continue
                # return

            img_path_test = os.path.join(root_rgb, d_name)
            original_img_rgb = Image.open(img_path_test).convert('RGB')
            # data_transform = transforms.Compose([transforms.ToTensor()])

            plot_img = draw_objs(original_img_rgb,
                                 boxes=predict_boxes,
                                 classes=predict_classes,
                                 scores=predict_scores,
                                 masks=predict_mask,
                                 category_index=category_index,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)
            # plt.imshow(plot_img)
            # plt.show()

            # save_path = os.path.join(r'E:\Data\RGBD\nflg-height\res-resres\a', d_name)
            save_path = os.path.join(r'E:\QT\study\DMask\pic\detection2', d_name)
            # 保存预测的图片结果
            plot_img.save(save_path)


if __name__ == '__main__':
    main()

