import os

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

data_dir = os.path.expanduser('~/data/yt8m/video')
output_dir = os.path.expanduser('~/data/yt8m/annotation/')


def read_tfrecords_and_make_csv(tfrecord_filename):
    succeed, error = 0, 0

    ext = os.path.splitext(tfrecord_filename)[-1]
    output_filename = os.path.basename(tfrecord_filename).replace(ext, '.csv')
    output_path = os.path.join(output_dir, output_filename)

    wf = open(output_path, 'w')

    record_iterator = tf.compat.v1.io.tf_record_iterator(path=tfrecord_filename)
    for string_record in record_iterator:
        try:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            feature = example.features.feature
            id = feature['id'].bytes_list.value[0].decode('UTF-8')
            id_url = 'http://data.yt8m.org/2/j/i/{}/{}.js'.format(id[:2], id)
            id_response = requests.get(id_url)
            if id_response.status_code == 200:
                id_code = id_response.text.split('\",\"')[1].split('\");')[0]
                labels = np.array(feature['labels'].int64_list.value).astype(str)
                line = '{},{}'.format(str(id_code), '|'.join(labels))
                wf.write(line + '\n')
                succeed += 1
            else:
                raise AssertionError
        except Exception as e:
            error += 1
    wf.close()

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

    # Parallel(n_jobs=multiprocessing.cpu_count())(delayed(read_tfrecords_and_make_csv)(tfrecord_path) for tfrecord_path in tqdm(tfrecord_paths))
