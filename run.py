import argparse

import torch

from dataset import VideoClassificationData
from utils import download_model, load_labels


def run(args):
    if args.model is None:
        download_model()
        args.model = 'model.pth'

    model = torch.load(args.model)
    model = model.to(args.device)
    model = model.eval()

    kinetics_id_to_classname = load_labels(args.labels)

    dataloader = VideoClassificationData()

    video_data = dataloader(args.video)
    inputs = [i.to(args.device)[None, ...] for i in video_data["video"]]

    preds = model(inputs)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices

    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
    print("Predicted labels: %s" % ", ".join(pred_class_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', default='model.pth', type=str, help='path to model')
    parser.add_argument('--labels', default='labels.csv', type=str, help='path to labels file')
    parser.add_argument('--video', default='data/sample.mp4', type=str, help='path to video file')
    parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
    args = parser.parse_args()

    run(args)
