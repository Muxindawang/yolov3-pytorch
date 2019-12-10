# -*- coding: utf-8 -*-
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def create_modules(module_defs):
    """
    module_defs是解析的yolov3.cfg文件
    [net] .....   [convolutional] ....
    Constructs module list of layer blocks from module configuration in module_defs
    根据module-defs 创建网络结构模型
    net:parameter 中保存的是训练参数的信息  不用于前向传播
    返回搭建的模型和超参数
    """
    hyperparams = module_defs.pop(0)    # 移除第一个元素 并返回该元素值 用于保存超参数
    output_filters = [int(hyperparams["channels"])]    # 第一层的filter:3
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        # 下面的代码表示 对于每种类型的网络层 如何选择 如何添加
        # 卷积层
        # [convolutional]
        # batch_normalize = 1
        # filters = 32/size = 3/stride = 1/pad = 1/activation = leaky
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                "conv_{}".format(module_i),
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_{}".format(module_i), nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_{}".format(module_i), nn.LeakyReLU(0.1))
        # 最大池化层  yolov3中不存在最大池化层
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module("_debug_padding_{}".format(module_i), nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_{}".format(module_i), maxpool)
        # 上采样
        elif module_def["type"] == "upsample":
            # 这里的module_def["stride"] 为上采样的缩放因子
            # 由于torch.nn.upsample被取消 所以定义一个upsample类
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_{}".format(module_i), upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            # output_filters[1:][1]+[2]或者output_filters[1:][1]
            # 此处的[1:][i]是指 output_filters从第二个到最后组成的列表中的第i个
            # 两个特征图相加
            modules.add_module("route_{}".format(module_i), EmptyLayer())
        # shortcut
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module("shortcut_{}".format(module_i), EmptyLayer())

        elif module_def["type"] == "yolo":
            '''
            [yolo]
            mask = 6,7,8
            anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            classes=80
            num=9
            jitter=.3
            ignore_thresh = .7
            truth_thresh = 1
            random=1'''
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]    # anchors列表中的第6,7,8个对于不同的yolo检测层选取不同尺寸的anchors
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(module_i), yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        # 在module-list(nn.ModuleList()创建) 的列表中 加入 modules(nn.Sequential())  即创建网络结构
        output_filters.append(filters)     # [3, 32, 64, 32, 64, 64, 128, ...]

    return hyperparams, module_list
    # hyperparams:::::
    # {'burn_in': '1000', 'max_batches': '500200', 'policy': 'steps', 'exposure': '1.5', 'subdivisions': '1',
    # 'width': '416', 'momentum': '0.9', 'saturation': '1.5', 'hue': '.1', 'type': 'net', 'batch': '16',
    # 'channels': '3', 'height': '416', 'learning_rate': '0.001', 'steps': '400000,450000', 'scales': '.1,.1',
    # 'decay': '0.0005', 'angle': '0'}


class Upsample(nn.Module):
    """ nn.Upsample 移除  定义Upsample类,forward用来实现上采样"""

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        # nms阈值
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.obj_scale = 1
        self.noobj_scale = 100

        self.metrics = {}

        self.img_dim = img_dim    # 输入图片尺寸
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        # 计算网格偏移
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size  # 缩小多少倍 图片尺寸/网格数量 下采样数量  416/13=32=2^5

        # Calculate offsets for each grid
        # grid_x， grid_y（1, 1, gride_size, gride_size）
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        # 图片缩小多少倍，对应的anchors也要缩小相应倍数 这样才能放到下采样后的网格中 13*13 26*26 52*52
        self.scaled_anchors = FloatTensor([(a_w/self.stride, a_h/self.stride) for a_w, a_h in self.anchors])

        # scaled_anchors shape（3,2）,3个anchors,每个anchor有w,h两个量.下面步骤是把这两个量划分开
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))  # （1,3, 1,1）
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))  # （1,3,1,1）

    def forward(self, x, targets=None, img_dim=None):
        # 计算总损失 以及 预测结果outputs  targets为真实边界框  用于计算ap recall等
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim  # 图片尺寸
        num_samples = x.size(0)  # (img_batch)
        grid_size = x.size(2)  # (feature_map_size)
        # x.shape = tensor([batch_size,num_anchors*(num_classes+5),grid_size,grid_size])
        # (batch_size, 255, grid_size, grid_size)
        # x就是最终输出的预测结果 255 = (80 + 4 + 1)* 3
        # 13*13*255
        prediction = (
            x.view(num_samples, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # print prediction.shape (batch_size, num_anchors, grid_size, grid_size, 85)

        # Get outputs
        # 这里的prediction是初步的所有预测，在grid_size*grid_size个网格中，它表示每个网格都会有num_anchor（3）个anchor框
        # x,y,w,h, pred_conf的shape都是一样的 (batch_size, num_anchor, gride_size, grid_size)
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf置信度   应该是[objectness*class]不是特别确定
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. (batch_size, num_anchor, gride_size, grid_size, cls)

        # If grid size does not match current we compute new offsets
        # print grid_size, self.grid_size
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # print self.grid_x, self.grid_y, self.anchor_w, self.anchor_h
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        # 这里是创建一个同等shape的tensor
        # 针对每个网格的偏移量,每个网格的单位长度为1,而预测的中心点（x,y）是归一化的（0,1之间）,所以可以直接相加
        # 广播机制
        pred_boxes[..., 0] = x.data + self.grid_x  # （batch_size, 1, gride_size, gride_size）
        # pred_boxes.shape = tensor.size([1,3,13,13])
        # 详细解析上一步是什么意思,首先看维度   x的维度13*13*1  什么意思  就是每个网格中都包含一个预测的x值
        #   那么距离左上角的距离就是   第一个网格左上角就是整个的左上角所以 +0  以此类推 +1 +2 +3 ...
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # # （1,3,1,1）
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        # anchor_w 是预先设定的anchor尺寸   w.data是预测的边界框的宽
        # 0 , 1   是指预测的中心点相对于图片左上角的偏移量
        # pred_boxes.shape = tensor.size([batch_size, num_anchors,grid_size,grid_size, 4])
        output = torch.cat(
            (
                # (batch_size, num_anchors*grid_size*grid_size, 4)
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # 放大到最初输入的尺寸
                # (batch_size, num_anchors*grid_size*grid_size, 1)
                pred_conf.view(num_samples, -1, 1),
                # (batch_size, num_anchors*grid_size*grid_size, 80)
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        # output.shape = tensor.size([batch_size, num_anchors*grid_size*grid_size, 85])
        if targets is None:
            # targets 是指ground truth
            return output, 0
        else:
            # pred_boxes => (batch_size, anchor_num, gride, gride, 4)
            # pred_cls => (batch_size, anchor_num, gride, gride, 80)
            # targets => (num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
            # scaled_anchors => (3, 2)
            # print pred_boxes.shape, pred_cls.shape, targets.shape, self.scaled_anchors.shape
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            #
            # iou_scores：预测框pred_boxes中的正确框与目标实体框target_boxes的交集IOU,以IOU作为分数,IOU越大,分值越高.
            # class_mask：将预测正确的标记为1（正确的预测了实体中心点所在的网格坐标，哪个anchor框可以最匹配实体，以及实体的类别）
            # obj_mask：将目标实体框所对应的anchor标记为1，目标实体框所对应的anchor与实体一一对应的
            # noobj_mask：将所有与目标实体框IOU小于某一阈值的anchor标记为1
            # tx, ty, tw, th： 需要拟合目标实体框的坐标和尺寸
            # tcls：目标实体框的所属类别
            # tconf：所有anchor的目标置信度

            # 这里计算得到的iou_scores，class_mask，obj_mask，noobj_mask，tx, ty, tw, th和tconf都是（batch, anchor_num, gride, gride）
            # 预测的x,y,w,h,pred_conf也都是（batch, anchor_num, gride, gride）

            # tcls 和 pred_cls 都是（batch, anchor_num, gride, gride，num_class）

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)

            # 坐标和尺寸的loss计算：
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            # anchor置信度的loss计算：
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  # tconf[obj_mask] 全为1
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])  # tconf[noobj_mask] 全为0
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # 类别的loss计算
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

            # loss汇总
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics 指标
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf

            obj_mask = obj_mask.float()

            # print type(iou50), type(detected_mask), type(conf50.sum()), type(iou75), type(obj_mask)
            #
            # print iou50.dtype, detected_mask.dtype, conf50.sum().dtype, iou75.dtype, obj_mask.dtype
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size

        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                # route层
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
                # shortcut层 和前面的层进行相加
            elif module_def["type"] == "yolo":
                # yolo layer forward return outputs loss
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path
        解析并加载weights-path中的权重文件'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values前五个是标题值
            self.header_info = header  # Needed to write header when saving weights 存储权重时候需要写进header
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            # 是否预训练
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                # 卷积层则下载权重
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    # 下载权重
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


# if __name__ == '__main__':



