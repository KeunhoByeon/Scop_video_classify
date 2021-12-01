import torch.nn as nn
from torchvision.models.resnet import resnet18


class VideoClassificationModel(nn.Module):
    def __init__(self, args):
        super(VideoClassificationModel, self).__init__()
        self.resnet18 = resnet18(pretrained=args.pretrained)
        self.lstm = nn.LSTM(750, 100)
        self.fc = nn.Linear(100 * 50, 2)

    def forward(self, x):
        features = self.resnet18(x)
        out = self.lstm(features)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out