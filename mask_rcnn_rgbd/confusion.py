import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from my_dataset_coco import CocoDetection

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone

from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    #定义一个全零的矩阵，作为混淆矩阵
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
    
    #p预测值和t真实标签
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy准确率
        sum_TP = 0
        for i in range(self.num_classes):#便利每个类别
            sum_TP += self.matrix[i, i] #统计对角线上元素的和
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity 精确率，召回率，特异度
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP #一行中除了TP求和
            FN = np.sum(self.matrix[:, i]) - TP #一列中除了TP求和
            TN = np.sum(self.matrix) - TP - FP - FN #矩阵求和减去TP ，FP ，FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0. #round表示只取3位小数
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


# def create_model(num_classes, load_pretrain_weights=True):
#     # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
#     backbone = resnet50_fpn_backbone(pretrain_path="resnet50_cut.pth", trainable_layers=3)
#     # backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)
#     model = MaskRCNN(backbone, num_classes=num_classes)

#     if load_pretrain_weights:
#         # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
#         weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco_cut.pth", map_location="cpu")
#         # weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")


#         for k in list(weights_dict.keys()):
#             if ("box_predictor" in k) or ("mask_fcn_logits" in k):
#                 del weights_dict[k]

#         print(model.load_state_dict(weights_dict, strict=False))

#     return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #要保持与训练时相同的预处理方式
    # data_transform = transforms.Compose([transforms.Resize(256),
    #                                      transforms.CenterCrop(224),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    batch_size = 16
    num_classes = 4 #除了背景的类
    box_thresh = 0.7

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }
    data_root = './data/coco2022'
    val_dataset = CocoDetection(data_root, "val", data_transform["val"])

    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=0,
                                                  )
    # validate_loader = torch.utils.data.DataLoader(val_dataset,
    #                                               batch_size=1,
    #                                               shuffle=False,
    #                                               pin_memory=True,
    #                                               num_workers=0,
    #                                               collate_fn=val_dataset.collate_fn)
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform)

    
    # validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=batch_size, shuffle=False,
    #                                               num_workers=2)
    # net = MobileNetV2(num_classes=5)


    net = create_model(num_classes=num_classes + 1)

    # load pretrain weights
    model_weight_path = "./save_weights/rgbd2/model_199.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)

    model_weight = torch.load(model_weight_path, map_location='cpu')
    model_weight = model_weight['model'] if 'model' in model_weight else model_weight
    net.load_state_dict(model_weight)
    print(model_weight)

    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net.to(device)

    # read class_indict
    json_label_path = './coco91_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=4, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()

