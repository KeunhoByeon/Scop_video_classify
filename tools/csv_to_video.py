import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

data_dir = os.path.expanduser('~/data/yt8m/annotation')
output_dir = os.path.expanduser('~/data/yt8m/mp4/')


def csv_2_video(csv_path):
    ext = os.path.splitext(csv_path)[-1]
    output_frames_dir = os.path.join(output_dir, os.path.basename(csv_path).replace(ext, ''))
    os.makedirs(output_frames_dir, exist_ok=True)

    id_list = []
    with open(csv_path, 'r') as rf:
        for line in rf.readlines():
            line_split = line.replace('\n', '').split(',')
            id_list.append(line_split[1])

    def download_video_from_id(id):
        url = 'https://www.youtube.com/watch?v={}'.format(id)
        try:
            urllib.request.urlretrieve(url, os.path.join(output_frames_dir, '{}.mp4'.format(id)))
        except Exception as e:
            return False
        return True

    with ThreadPoolExecutor(max_workers=128) as pool:
        succeed_list = np.array(list(pool.map(download_video_from_id, id_list)))

    succeed = np.sum(succeed_list)
    error = len(succeed_list) - succeed

    del id_list
    del succeed_list

    return succeed, error


if __name__ == '__main__':
    np.random.seed(1003)

    csv_paths = []
    for path, dir, files in os.walk(data_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext == '.csv':
                csv_paths.append(os.path.join(path, filename))
    csv_paths = np.array(csv_paths)
    np.random.shuffle(csv_paths)

    total_succeed, total_error = 0, 0
    for csv_path in tqdm(csv_paths):
        succeed, error = csv_2_video(csv_path)
        total_succeed += succeed
        total_error += error

    print('Succeed: {}\nError: {}'.format(total_succeed, total_error))
