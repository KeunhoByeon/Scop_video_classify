import argparse

import torch
from tqdm import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
from dataset import VideoClassificationData
from utils import download_model, load_labels, get_topk_classes, clear_memory


def run(args):
    with torch.no_grad():
        if args.device is None:
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if args.model is None:
            download_model()
            args.model = 'model.pth'

        model = torch.load(args.model)
        model = model.to(args.device)
        model = model.eval()

        labels = load_labels(args.labels)

        dataloader = VideoClassificationData()
        video = EncodedVideo.from_path(args.video)
        duration = max(video.duration, args.clip_duration)
        num_clips = min(int(duration / args.clip_duration), args.max_clips)
        start_clip = 0

        preds = torch.zeros((1, 400), device=args.device)
        for i in tqdm(range(start_clip, num_clips)):
            clear_memory()

            input = dataloader(video, start_sec=i * args.clip_duration, end_sec=(i + 1) * args.clip_duration)
            input = [i.to(args.device)[None, ...] for i in input["video"]]
            preds += model(input) / num_clips

        pred_classes = get_topk_classes(preds, topk=5)
        pred_class_names = [labels[int(i)] for i in pred_classes[0]]
        print("Predicted labels: %s" % ", ".join(pred_class_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', default='model.pth', type=str, help='path to model')
    parser.add_argument('--labels', default='labels.csv', type=str, help='path to labels file')
    parser.add_argument('--video', default='data/eating.mp4', type=str, help='path to video file')
    parser.add_argument('--device', default=None, help='cpu or cuda (None for auto)')
    parser.add_argument('--clip_duration', default=16, type=int)
    parser.add_argument('--max_clips', default=5, type=int)
    args = parser.parse_args()

    run(args)
