import os
import torch
from torchvision.io import write_video
import numpy as np

class recorder:
    def __init__(self, video_path):
        self.video_path = video_path
        if self.video_path is not None:
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)

    def init(self, name, size=256, fps=30, enabled=True):
        if enabled:
            self.video_name = name
            self.video_size = size
            self.video_fps = fps
            self.frames = []
            
    def record(self, image):
        if hasattr(self, 'frames'):
            self.frames.append(torch.tensor(np.fliplr(image).copy())) #不再需要transpose(1,2,0)

    def release(self, name):
        self.video_name = name
        if hasattr(self, 'frames'):
            if self.video_path is not None:
                video_tensor = torch.stack(self.frames)
                video_path = os.path.join(self.video_path, self.video_name)
                write_video(video_path, video_tensor, self.video_fps)