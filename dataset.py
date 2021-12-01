import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (CenterCropVideo, NormalizeVideo, )


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha=4):
        super().__init__()

        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(frames, 1, torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long(), )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class VideoClassificationData():
    def __init__(self, side_size=256, crop_size=256, num_frames=32, sampling_rate=2, frames_per_second=30, alpha=4):
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]

        self.side_size = side_size
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frames_per_second = frames_per_second
        self.alpha = alpha

        self.start_sec = 0
        self.clip_duration = (self.num_frames * self.sampling_rate) / self.frames_per_second
        self.end_sec = self.start_sec + self.clip_duration

        applying_transforms = [UniformTemporalSubsample(self.num_frames),
                               Lambda(lambda x: x / 255.0),
                               NormalizeVideo(self.mean, self.std),
                               ShortSideScale(size=self.side_size),
                               CenterCropVideo(self.crop_size),
                               PackPathway(alpha=self.alpha)]

        self.transform = ApplyTransformToKey(key="video", transform=Compose(applying_transforms), )

    def __call__(self, video_path):
        video = EncodedVideo.from_path(video_path)

        # Load the desired clip
        video_data = video.get_clip(start_sec=self.start_sec, end_sec=self.end_sec)

        # Apply a transform to normalize the video input
        video_data = self.transform(video_data)

        return video_data
