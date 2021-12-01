import base64
import os
from io import BytesIO

import urllib
import boto3
import numpy as np
import torch
from PIL import Image


def get_state_dict_from_aws(bucket='scop-bai', key='pytorch_models/Favorfit/model_state_dict_01.pth', tmp_model_path='/tmp/tmp_model.pth', use_new=False):
    if not os.path.isfile(tmp_model_path):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket)
        bucket.download_file(key, tmp_model_path)
    else:
        print('Load existing state dict file')

    return torch.load(tmp_model_path)


def decode_url(url):
    return urllib.parse.unquote(url)


def b64_to_image(b64_image):
    b64_image = b64_image.split(',')[1]
    decoded_img = base64.b64decode(b64_image)
    img = BytesIO(decoded_img)
    img = Image.open(img)
    return img


def pil2tensor(ori_img, input_size=224):
    # 1) Resize and padding
    old_size = ori_img.size
    ratio = float(input_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    ori_img = ori_img.resize(new_size, Image.ANTIALIAS)

    img = Image.new("RGB", (input_size, input_size))
    img.paste(ori_img, ((input_size - new_size[0]) // 2, (input_size - new_size[1]) // 2))

    # 2) Make hsv image
    hsv = img.copy().convert('HSV')
    hsv = (np.array(hsv).astype(np.float32) / 255.)

    # 3) Convert as tensor shape
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    mean = np.flip(mean)
    std = np.flip(std)

    img = (np.array(img).astype(np.float32) / 255.)
    img = (img - mean) / std

    # 4) Stack rgb and hsv
    input_image = np.dstack((img, hsv))
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.array([input_image])
    input_image = torch.from_numpy(input_image)

    return input_image


def extract_output(model, img):
    # TODO: We need to change model
    with torch.no_grad():
        output, features = model(img)
        return features


def extract_similarity(data_arr):
    data_arr = np.array(data_arr)
    data_arr = data_arr / np.linalg.norm(data_arr, axis=1)[:, None]
    matrix = np.einsum('ik,jk->ij', data_arr, data_arr)
    return matrix
