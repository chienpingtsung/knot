import logging
import random
from itertools import count

import numpy as np
from easydict import EasyDict
from torch.utils.data import Dataset

from lib.dataset.base import VideoDataset
from lib.util.config import hyper

logger = logging.getLogger(__name__)


class VideoDatasetProxy(Dataset):
    def __init__(self, datasets, processing=None):
        super().__init__()

        self.datasets = datasets
        self.processing = processing

        self.weights = hyper.weights if hyper.weights else [len(d) for d in datasets]
        self.num_template = hyper.num_template
        self.num_search = hyper.num_search
        self.max_interval = hyper.max_interval
        self.manner = hyper.manner

    def __len__(self):
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
        return total

    def __getitem__(self, item):
        for c in count():
            if c > 100:
                logger.warning('Data sampling has been over 100 loops.')

            dataset = random.choices(self.datasets, self.weights)[0]
            video_id, anno, valid, visible = self.sample_video_id(dataset)

            if isinstance(dataset, VideoDataset):
                sample_func = {'causal': lambda *args, **kwargs: self.causal_manner_sample(*args, **kwargs),
                               'trident': lambda *args, **kwargs: self.trident_manner_sample(*args, **kwargs)}
                template_ids, search_ids = sample_func[self.manner](np.ones_like(valid))
            else:
                template_ids = [0] * self.num_template
                search_ids = [0] * self.num_search

            template_frames, template_anno = dataset.get_frames(video_id, template_ids, anno)
            search_frames, search_anno = dataset.get_frames(video_id, search_ids, anno)

            data = EasyDict({'template_frames': template_frames,
                             'template_bbox': template_anno.groundtruth,
                             'template_visible': (valid & visible)[template_ids, ...],
                             'search_frames': search_frames,
                             'search_bbox': search_anno.groundtruth,
                             'search_visible': (valid & visible)[search_ids, ...],
                             'nlp': template_anno.nlp})

            if self.processing:
                try:
                    data = self.processing(data)
                except:
                    continue

            return data

    def sample_video_id(self, dataset):
        for c in count():
            if c > 100:
                logger.warning('Video id sampling has been over 100 loops.')

            video_id = random.randrange(len(dataset))
            anno = dataset.get_annotation(video_id)

            valid = (anno.groundtruth[..., 2] > 0) & (anno.groundtruth[..., 3] > 0)
            visible = 1 - (anno.occlusion | anno.out_of_view)

            if visible.sum() > 2 * (self.num_template + self.num_search) and len(visible) > 20 or \
                    not isinstance(dataset, VideoDataset):
                return video_id, anno, valid, visible

    def sample_frame_ids(self, condition, num=1, start=None, stop=None):
        if start is None or start < 0:
            start = 0
        if stop is None or stop > len(condition):
            stop = len(condition)

        optional_ids = [i for i in range(start, stop) if condition[i]]

        if optional_ids:
            return random.choices(optional_ids, k=num)
        return None

    def causal_manner_sample(self, condition):
        for c in count(0, 5):
            if c > 1000:
                logger.warning('Interval has been over 1000 frames.')

            base_template = self.sample_frame_ids(condition, num=1,
                                                  start=self.num_template - 1,
                                                  stop=len(condition) - self.num_search)[0]
            prev_templates = self.sample_frame_ids(condition, num=self.num_template - 1,
                                                   start=base_template - self.max_interval - c)
            if prev_templates is None:
                continue

            searches = self.sample_frame_ids(condition, num=self.num_search,
                                             start=base_template + 1,
                                             stop=base_template + 1 + self.max_interval + c)
            if searches is None:
                continue

            prev_templates.append(base_template)
            return sorted(prev_templates), sorted(searches)

    def trident_manner_sample(self, condition):
        for c in count():
            if c > 100:
                logger.warning('Frame sampling has been over 100 loops.')

            static_template = self.sample_frame_ids(condition, num=1)[0]
            search = self.sample_frame_ids(condition, num=1)[0]

            if static_template < search:
                start, stop = max(static_template, search - self.max_interval), search
            else:
                start, stop = search, min(static_template, search + self.max_interval)

            online_template = self.sample_frame_ids(condition, num=1, start=start, stop=stop)
            if online_template is None:
                continue

            return [static_template, online_template[0]], [search]

def collate_fn():
    pass
