import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

data_dir = os.path.expanduser('~/data/yt8m/video')
output_dir = os.path.expanduser('~/data/yt8m/annotation/')


def get_id_from_url(id_url):
    try:
        id_response = requests.get(id_url)
        if id_response.status_code == 200:
            return id_response.text.split('\",\"')[1].split('\");')[0]
        else:
            raise AssertionError
    except Exception as e:
        return None


def read_tfrecords_and_make_csv(tfrecord_filename):
    succeed, error = 0, 0

    ext = os.path.splitext(tfrecord_filename)[-1]
    output_filename = os.path.basename(tfrecord_filename).replace(ext, '.csv')
    output_path = os.path.join(output_dir, output_filename)

    temp_id_list, id_url_list, labels_list = [], [], []
    record_iterator = tf.compat.v1.io.tf_record_iterator(path=tfrecord_filename)
    for string_record in record_iterator:
        try:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            feature = example.features.feature
            temp_id = feature['id'].bytes_list.value[0].decode('UTF-8')
            id_url = 'http://data.yt8m.org/2/j/i/{}/{}.js'.format(temp_id[:2], temp_id)
            labels = np.array(feature['labels'].int64_list.value).astype(str)

            temp_id_list.append(temp_id)
            id_url_list.append(id_url)
            labels_list.append(labels)

        except Exception as e:
            error += 1
    del record_iterator

    with ThreadPoolExecutor(max_workers=32) as pool:
        id_list = list(pool.map(get_id_from_url, id_url_list))

    with open(output_path, 'w') as wf:
        for temp_id, id, labels in zip(temp_id_list, id_list, labels_list):
            if id is None:
                error += 1
                continue

            wf.write('{},{},{}\n'.format(str(temp_id), str(id), '|'.join(labels)))
            succeed += 1
    del temp_id_list
    del id_url_list
    del labels_list

    return succeed, error


if __name__ == '__main__':
    tfrecord_paths = []
    for path, dir, files in os.walk(data_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext == '.tfrecord':
                tfrecord_paths.append(os.path.join(path, filename))

    total_succeed, total_error = 0, 0
    for tfrecord_path in tqdm(tfrecord_paths):
        succeed, error = read_tfrecords_and_make_csv(tfrecord_path)
        total_succeed += succeed
        total_error += error

    print('Succeed: {}\nError: {}'.format(total_succeed, total_error))
