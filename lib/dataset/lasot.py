import os

import numpy as np
import pandas
from PIL import Image
from easydict import EasyDict

from lib.dataset.base import VideoDataset


class LaSOT(VideoDataset):
    def __init__(self, root, partition):
        super().__init__('LaSOT', root)

        self.video_list = self.build_video_list(partition)

    def read_partition(self, path):
        return pandas.read_csv(path, header=None).squeeze().to_numpy().tolist()

    def read_full_occlusion(self, path):
        return pandas.read_csv(path, header=None).squeeze().to_numpy()

    def read_groundtruth(self, path):
        return pandas.read_csv(path, header=None).to_numpy()

    def read_nlp(self, path):
        return pandas.read_csv(path, header=None).squeeze()

    def read_out_of_view(self, path):
        return pandas.read_csv(path, header=None).squeeze().to_numpy()

    def build_video_list(self, partition):
        class_list = [c for c in os.listdir(self.root)]
        partition = self.read_partition(partition) if partition else None

        video_list = []
        for c in class_list:
            for v in os.listdir(self.root.joinpath(c)):
                if partition is None or v in partition:
                    video_list.append(self.root.joinpath(c, v))

        return video_list

    def get_frames(self, video_id, frame_ids, anno=None):
        video_path = self.video_list[video_id].joinpath('img')

        frames = [Image.open(video_path.joinpath(f'{id + 1:08}.jpg')) for id in frame_ids]

        if anno is None:
            anno = self.get_annotation(video_id)

        frames_anno = EasyDict()
        for key, value in anno.items():
            if isinstance(value, np.ndarray):
                frames_anno[key] = value[frame_ids, ...].copy()
            else:
                frames_anno[key] = value

        return frames, frames_anno

    def get_annotation(self, video_id):
        video_path = self.video_list[video_id]

        occlusion = self.read_full_occlusion(video_path.joinpath('full_occlusion.txt'))
        groundtruth = self.read_groundtruth(video_path.joinpath('groundtruth.txt'))
        nlp = self.read_nlp(video_path.joinpath('nlp.txt'))
        out_of_view = self.read_out_of_view(video_path.joinpath('out_of_view.txt'))

        return EasyDict({'occlusion': occlusion, 'groundtruth': groundtruth, 'nlp': nlp, 'out_of_view': out_of_view})
