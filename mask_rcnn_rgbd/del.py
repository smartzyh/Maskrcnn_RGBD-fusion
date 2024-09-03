import torch



# model_del = torch.load("./maskrcnn_resnet50_fpn_coco.pth")
# model_del = torch.load("./resnet50.pth")
# model_del = torch.load("./save_weights/rgbd/model_19.pth")
model_del = torch.load("./resnet101-5d3b4d8f.pth")
# model = model.state_dict()

print(model_del)

#删除预训练权重的第一层
del_key = []
for key,_ in model_del.items():
    if "conv1.weight" in key : 
        del_key.append(key)
        break
        # del model.state_dict[name]
        # break
        # del model.state_dict['conv1.weight']
for key in del_key:
    del model_del[key]

torch.save( model_del,'./resnet101.pth')
print('1111111111111111111111111')
for name in model_del:
    print(name)
