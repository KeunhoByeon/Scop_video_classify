import gc

import cv2
import torch
from PyQt5 import QtGui


def load_labels(file_path):
    kinetics_id_to_classname = {}
    with open(file_path, "r") as rf:
        for line in rf.readlines():
            id, classname = line.replace('\n', '').split(',')
            kinetics_id_to_classname[int(id)] = classname

    return kinetics_id_to_classname


def get_topk_classes(preds, topk=5):
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=topk).indices

    return pred_classes


def resize_and_pad_image(img, input_size=224):
    # 1) resize image
    old_size = img.shape[:2]
    ratio = float(input_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))

    # 2) pad image
    delta_w = input_size - new_size[1]
    delta_h = input_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img


def cv_to_pixmap(img, input_size):
    img = resize_and_pad_image(img, input_size=input_size)
    h, w, c = img.shape
    bpl = 3 * w
    qtimage = QtGui.QImage(img.data, w, h, bpl, QtGui.QImage.Format_RGB888)
    qtpixmap = QtGui.QPixmap(qtimage)
    return qtpixmap


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect(generation=0)
    gc.collect(generation=1)
    gc.collect(generation=2)
