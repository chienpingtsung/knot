import abc
from pathlib import Path


class VideoDataset(abc.ABC):
    def __init__(self, name, root):
        self.name = name
        self.root = Path(root)

        self.video_list = []

    def __len__(self):
        return len(self.video_list)

    @abc.abstractmethod
    def get_frames(self, video_id, frame_ids, anno=None):
        pass

    @abc.abstractmethod
    def get_annotation(self, video_id):
        pass
