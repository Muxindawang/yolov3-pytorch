# -*- coding: utf-8 -*-
from __future__ import division

from models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

# from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
    # 梯度累积
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # 默认的网络结构文件  多少层 每层什么结构 参数
    '''
    [convolutional]
    batch_normalize=1
    filters=32
    size=3
    stride=1
    pad=1
    activation=leaky
    '''
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # data config
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    # 如果指定就从检查点开始
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # 保存模型权重的间隔
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    # 测试集上的间隔计算
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    # 计算map
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    # 多尺度训练
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    # logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)    # 导入coco.data
    """
    classes= 80
    train=data/coco/trainvalno5k.txt
    valid=data/coco/5k.txt
    names=data/coco.names
    backup=backup/
    eval=coco
    """
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])    # 下载类别 path=data/coco.names

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))   # 加载模型权重 保存的文件后缀是 pt 或 pth
        else:
            model.load_darknet_weights(opt.pretrained_weights)   # 家在模型权重

    # model = torch.nn.DataParallel(model).cuda()

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    # 优化器
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    # 迭代开始
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (img_pth, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            # img: (batch_size, channel, height, width)
            # target: (num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            # 展示训练进度
            # ---- [Epoch 1/5, Batch 1/58632] ----
            # metric    "YOLO Layer 0"   "YOLO Layer 1"   "YOLO Layer 2"
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            metric_table = [["Metrics", *["YOLO Layer {}".format(i) for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                # 从metrics中获取指标类型保存到formats中
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                # 解析yolo层的参数放进tensorboard_log中
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [("{}_{}".format(name, j+1), metric)]
                            # ('loss_1', 77.54456454)
                tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # log_str += AsciiTable(metric_table).table
            log_str += "\nTotal loss {}".format(loss.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += "\n---- ETA {}".format(time_left)

            print(log_str)

            model.seen += imgs.size(0)

            # if batch_i > 10:
            #     break

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            # print(AsciiTable(ap_table).table)
            print("---- mAP {}".format(AP.mean()))

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_%d.pth" % epoch)
