# coding:utf8
from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    """包含卷积层BN层的话就初始化数据shape """
    classname = m.__class__.__name__   # 获取类名
    if classname.find("Conv") != -1:     # find函数不包含会返回-1
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    # prediction   current_dim = 416   samples
    """
    Rescales bounding boxes to the original shape
    用于detect.py中   在416*416图片中画完boundingbox之后 将图片大小复原到原尺寸 坐标变化 相对于左上角距离
    长 2 正  长边坐标(w长则对应x坐标,l1到左边界限距离)  l1*416/max(h,w)   短边坐标   l2*416/max(h,w)+(416-l2*416/max(h,w))/2
    """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    # max(original_shape)是指扩充的倍数
    # 如果h大就扩充x  w大就扩充y
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_w = current_dim - pad_x
    unpad_h = current_dim - pad_y
    # Rescale bounding boxes to dimension of original image ?????????
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    """x[x,y,w,h]变为x1,y1,x2,y2"""
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.

    precision = tp / (tp + fp)    ap是预测对的/所有预测出来的目标
    recall = tp / (tp + fn) recall 是预测对的/所有的目标(gt)
    ap 是 P-r曲线的面积
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects  真实的目标  recall的分母
        n_p = i.sum()  # Number of predicted objects   预测出来的目标  ap的分母

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    # rp曲线   recall从0-1
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # 找到各个阶段的最大precision值

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # (recall[i+1] - recall[i])*max(precision when recall>=recall[i+1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []

    # print "outputs len: {}".format(len(outputs))
    # print "targets shape: {}".format(targets.shape)
    # outputs: (batch_size, pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
    # target:  (num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        # output: (pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
        output = outputs[sample_i]
        # print "output: {}".format(output.shape)

        pred_boxes = output[:, :4]  # 预测框的x,y,w,h
        pred_scores = output[:, 4]  # 预测框的置信度
        pred_labels = output[:, -1]  # 预测框的类别label

        # 长度为pred_boxes_num的list，初始化为0，如果预测框和实际框匹配，则设置为1
        true_positives = np.zeros(pred_boxes.shape[0])

        # 获得真实目标框的类别label
        # annotations = targets[targets[:, 0] == sample_i][:, 1:]
        annotations = targets[targets[:, 0] == sample_i]
        annotations = annotations[:, 1:] if len(annotations) else []
        target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):  # len(annotations)>0: 表示这张图片有真实的目标框
            detected_boxes = []
            target_boxes = annotations[:, 1:]  # 真实目标框的x,y,w,h

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                # 如果该预测框的类别标签不存在与目标框的类别标签集合中，则必定是预测错误
                if pred_label not in target_labels:
                    continue

                # 将一个预测框与所有真实目标框做IOU计算，并获取IOU最大的值(iou)，和与之对应的真实目标框的索引号(box_index)
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # 如果最大IOU大于阈值，则认为该真实目标框被发现。注意要防止被重复记录
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1  # 对该预测框设置为1
                    detected_boxes += [box_index]  # 记录被发现的实际框索引号，防止预测框重复标记，即一个实际框只能被一个预测框匹配
        # 保存当前图片被预测的信息
        # true_positives：预测框的正确与否，正确设置为1，错误设置为0
        # pred_scores：预测框的x,y,w,h
        # pred_labels：预测框的类别标签
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    # print w1, w2, h1, h2

    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    # print inter_area, union_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # prediction: (batch_size, num_anchors*grid_size*grid_size*3, 85) 85 => (x,y,w,h, conf, cls)
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # 得到置信预测框：过滤anchor置信度小于阈值的预测框
        # print image_pred.shape (num_anchors*grid_size*grid_size*3, 85) 85 => (x,y,w,h, conf, cls)
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]   # confidence
        # print image_pred.shape  (more_than_conf_thres_num, 85) 85 => (x,y,w,h, conf, cls)
        # 先筛选掉置信度(objecness)小的

        # If none are remaining => process next image
        # 基于anchor的置信度过滤完后，看看是否还有保留的预测框，如果都被过滤，则认为没有实体目标被检测到
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        # 计算处理：先选取每个预测框所代表的最大类别值，再将这个值乘以对应的anchor置信度，这样将类别预测精准度和置信度都考虑在内。
        # 每个置信预测框都会对应一个score值
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        # 基于score值，将置信预测框从大到小进行排序
        # image_pred = image_pred[(-score).argsort()]
        # 置信预测：image_pred ==》(more_than_conf_thres_num, 85) 85 => (x,y,w,h, conf, cls)
        image_pred = image_pred[torch.sort(-score, dim=0)[1]]
        # image_pred[:, 5:] ==> (more_than_conf_thres_num, cls)
        # 该处理是获取每个置信预测框所对应的类别预测分值（class_confs）和类别索引（class_preds）
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # 将置信预测框的 x,y,w,h,conf，类别预测分值和类别索引关联到一起
        # detections ==》 (more_than_conf_thres_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # detections[0, :4]是第一个置信预测框，也是当前序列中分值最大的置信预测框
            # 计算当前序列的第一个（分值最大）置信预测框与整个序列预测框的IOU，并将IOU大于阈值的设置为1，小于的设置为0。
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # 匹配与当前序列的第一个（分值最大）置信预测框具有相同类别标签的所有预测框（将相同类别标签的预测框标记为1）
            label_match = detections[0, -1] == detections[:, -1]

            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            # 与当前序列的第一个（分值最大）置信预测框IOU大，说明这些预测框与其相交面积大，
            # 如果这些预测框的标签与当前序列的第一个（分值最大）置信预测框的相同，则说明是预测的同一个目标，
            # 对与当前序列第一个（分值最大）置信预测框预测了同一目标的设置为1（包括当前序列第一个（分值最大）置信预测框本身）。
            invalid = large_overlap & label_match
            # 取出对应置信预测框的置信度，将置信度作为权重
            # invalid边界框 iou大 并且类别相同
            weights = detections[invalid, 4:5]

            # Merge overlapping bboxes by order of confidence
            # 把预测为同一目标的预测框进行合并，合并后认为是最优的预测框。合并方式如下：
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 保存当前序列中最终识别的预测框
            keep_boxes += [detections[0]]
            # ~invalid表示取反，将之前的0变为1，即选取剩下的预测框，进行新一轮的计算
            detections = detections[~invalid]
        if keep_boxes:
            # 每张图片的最终预测框有pred_boxes_num个，output[image_i]的shape：
            # (pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
            output[image_i] = torch.stack(keep_boxes)

    # (batch_size, pred_boxes_num, 7) 7 =》x,y,w,h,conf,class_conf,class_pred
    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    # pred_boxes => (batch_size, anchor_num, gride, gride, 4)
    # pred_cls => (batch_size, anchor_num, gride, gride, 80)
    # targets => (num, 6)  6=>(batch_index, cls, center_x, center_y, widht, height)
    # 真实边界框
    # anchors => (3, 2)

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # batch num
    nA = pred_boxes.size(1)  # anchor num
    nC = pred_cls.size(-1)  # class num => 80
    nG = pred_boxes.size(2)  # gride

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, anchor_num, gride, gride)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    # tx=(gt_X-ax)/aw -faster rcnn
    # tx = gtx - ax   -yolov3
    # gt_x目标真实中心x坐标 ax anchor中心点x坐标 aw anchor的宽度
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)  # (batch_size, anchor_num, gride, gride, class_num)

    # Convert to position relative to box
    # 这一步是将x,y,w,h这四个归一化的变量变为真正的尺寸，因为当前图像的尺寸是nG，所以乘以nG。
    # print target[:, 2:6].shape  # (num, 4)
    target_boxes = target[:, 2:6] * nG  # (num, 4)  4=>(center_x, center_y, widht, height)
    gxy = target_boxes[:, :2]  # (num, 2)
    gwh = target_boxes[:, 2:]  # (num, 2)
    # print target_boxes.shape, gxy.shape, gwh.shape

    # Get anchors with best iou
    # 这一步是为每一个目标框从三种anchor框中分配一个最优的.
    # anchor 是设置的锚框，gwh是真实标记的宽高，这里是比较两者的交集，选出最佳的锚框,因为只是选择哪种锚框，不用考虑中心坐标。
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # (3, num)
    # ious（3，num），该处理是为每一个目标框选取一个IOU最大的anchor框，best_ious表示最大IOU的值，best_n表示最大IOU对应anchor的index
    best_ious, best_n = ious.max(0)  # best_ious 和 best_n 的长度均为 num， best_n是num个目标框对应的anchor索引

    # Separate target values
    # .t() 表示转置，（num，2） =》（2，num）
    # （2，num）  2=>(batch_index, cls） =》 b(num)表示对应num个index, target_labels(num)表示对应num个labels
    # long去除小数点
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()  # gx表示num个x， gy表示num个y
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()  # .long()是把浮点型转为整型（去尾），这样就可以得到目标框中心点所在的网格坐标


    # ---------------------------得到目标实体框obj_mask和目标非实体框noobj_mask  start----------------------------
    # Set masks
    # 表示batch中的第b张图片，其网格坐标为(gj, gi)的单元网格存在目标框的中心点，该目标框所匹配的最优anchor索引为best_n
    obj_mask[b, best_n, gj, gi] = 1  # b是指第几个targets 对目标实体框中心点所在的单元网格，其最优anchor设置为1
    noobj_mask[b, best_n, gj, gi] = 0  # 对目标实体框中心点所在的单元网格，其最优anchor设置为0 （与obj_mask相反）

    # Set noobj mask to zero where iou exceeds ignore threshold
    # ious.t(): (3, num) => (num, 3)
    # 这里不同与上一个策略，上个策略是找到与目标框最优的anchor框，每个目标框对应一个anchor框。
    # 这里不考虑最优问题，只要目标框与anchor的IOU大于阈值，就认为是有效anchor框，即noobj_mask对应的位置设置为0
    # ious [3,num_iou]
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # 以上操作得到了目标实体框obj_mask和目标非实体框noobj_mask，目标实体框是与实体一一对应的，一个实体有一个最匹配的目标框；
    # 目标非实体框noobj_mask，该框既不是实体最匹配的，而且还要该框与实体IOU小于阈值，这也是为了让正负样例更加明显。
    # ---------------------------得到目标实体框obj_mask和目标非实体框noobj_mask  end------------------------------

    # ---------------------------得到目标实体框的归一化坐标（tx, ty, tw, th）  start------------------------------
    # Coordinates
    # 将x,y,w,h重新归一化，
    # 注意：要明白这里为什么要这么做，此处的归一化和传入target的归一化方式不一样，
    # 传入target的归一化是实际的x,y,w,h / img_size. 即实际x,y,w,h在img_size中的比例，
    # 此处的归一化中，中心坐标x,y是基于单元网络的，w,h是基于anchor框，此处归一化的x,y,w,h，也是模型要拟合的值。
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # tw=log(gtw/aw)
    # ---------------------------得到目标实体框的归一化坐标（tx, ty, tw, th）  end---------------------------------


    # One-hot encoding of label
    # 表示batch中的第b张图片，其网格坐标为(gj, gi)的单元网格存在目标框的中心点，该目标框所匹配的最优anchor索引为best_n，其类别为target_labels
    tcls[b, best_n, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    # class_mask:将预测正确的标记为1（正确的预测了实体中心点所在的网格坐标，哪个anchor框可以最匹配实体，以及实体的类别）
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # iou_scores：预测框pred_boxes中的正确框与目标实体框target_boxes的交集IOU，以IOU作为分数，IOU越大，分值越高。
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    # tconf：正确的目标实体框，其对应anchor框的置信度为1，即置信度的标签，这里转为float，是为了后面和预测的置信度值做loss计算。
    tconf = obj_mask.float()

    # iou_scores：预测框pred_boxes中的正确框与目标实体框target_boxes的交集IOU，以IOU作为分数，IOU越大，分值越高。
    # class_mask：将预测正确的标记为1（正确的预测了实体中心点所在的网格坐标，哪个anchor框可以最匹配实体，以及实体的类别）
    # obj_mask：将目标实体框所对应的anchor标记为1，目标实体框所对应的anchor与实体一一对应的
    # noobj_mask：将所有与目标实体框IOU小于某一阈值的anchor标记为1
    # tx, ty, tw, th： 需要拟合目标实体框的坐标和尺寸
    # tcls：目标实体框的所属类别
    # tconf：所有anchor的目标置信度
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
