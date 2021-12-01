import os
import shutil

import cv2
import pafy
import requests
import tensorflow as tf
import numpy as np
import torch


###############################################################################################################################
# Dataset Utils
###############################################################################################################################
def get_tfrecords_data(tfrecord_filename):
    examples = []
    error_ptr = 0

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        feature = example.features.feature
        id = feature['id'].bytes_list.value[0].decode('UTF-8')
        id_url = 'http://data.yt8m.org/2/j/i/{}/{}.js'.format(id[:2], id)
        id_response = requests.get(id_url)
        if id_response.status_code == 200:
            id_code = id_response.text.split('\",\"')[1].split('\");')[0]
            examples.append({'id': id_code, 'labels': feature['labels'].int64_list.value, 'mean_audio': feature['mean_audio'].float_list.value, 'mean_rgb': feature['mean_rgb'].float_list.value})
        else:
            error_ptr += 1

    # print(error_ptr)
    return examples


def load_video_frames(id_code, frame_shape=(320, 180), debug=False):
    url = "https://www.youtube.com/watch?v={}".format(id_code)
    cap = cv2.VideoCapture(pafy.new(url).getbest(preftype="mp4").url)

    frames = []
    ret, frame = cap.read()
    while ret:
        if frame_shape is not None:
            frame = resize_and_pad(frame, frame_shape)
        if debug:
            cv2.imshow('T', frame)
            key = cv2.waitKey(0)
            if key == 27:
                break
        ret, frame = cap.read()
        frames.append(frame)

    return frames


def resize_and_pad(img, input_size):
    # 1) get ratio
    old_size = img.shape[:2]
    ratio = min(float(input_size[0]) / old_size[0], float(input_size[1]) / old_size[1])

    # 2) resize image
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_img = cv2.resize(img, (new_size[1], new_size[0]))

    # 3) pad image
    delta_w = input_size[1] - new_size[1]
    delta_h = input_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    output = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return output



###############################################################################################################################
# Model Utils
###############################################################################################################################
def save_checkpoint(state, is_best, filename, best_filename, result_dir='results/'):
    os.makedirs(result_dir, exist_ok=True)
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(result_dir, filename), os.path.join(result_dir, best_filename))


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '{name} {val:.3f} (avg: {avg:.3f})'.format(name=self.name, val=self.val, avg=self.avg)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, log_path, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_path = log_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def write(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        with open(self.log_path, 'at') as wf:
            wf.write(str('\t'.join(entries)) + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).cpu().data.numpy()[0])
        return res


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
