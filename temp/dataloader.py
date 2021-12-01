import os

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from temp.utils import load_video_frames


class Youtube8MDataset(data.Dataset):
    def __init__(self, args, split, input_size=224):
        self.args = args
        self.split = split

        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.samples = []

        csv_paths = []
        for path, dir, files in os.walk(args.data):
            for filename in files:
                ext = os.path.splitext(filename)[-1].lower()
                if ext == '.csv' and self.split.lower() in filename.lower():
                    csv_paths.append(os.path.join(path, filename))

        for csv_path in tqdm(csv_paths):
            rf = open(csv_path, 'r')
            for line in rf.readlines():
                line_split = line.replace('\n', '').split(',')
                id = line_split[0]
                labels = np.array(line_split[1].split('|')).astype(int)
                self.samples.append({'id': id, 'labels': labels})
            rf.close()

        print('=> {}set loaded {} videos'.format(self.split, len(self.samples)))

    def __getitem__(self, index):
        sample = self.samples[index]

        id = sample['id']
        labels = sample['labels']
        # mean_audio = sample['mean_audio']
        # mean_rgb = sample['mean_rgb']

        raw_frames = load_video_frames(id, (self.input_size, self.input_size))
        frames = []
        for input_image in raw_frames:
            input_image = (input_image.astype(np.float32) / 255.)
            input_image = (input_image - self.mean) / self.std
            input_image = input_image.transpose(2, 0, 1)
            frames.append(input_image)
        frames = np.array(frames)

        return torch.from_numpy(frames), labels

    def __len__(self):
        return len(self.samples)
