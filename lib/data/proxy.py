import random
from itertools import count

from torch.utils.data import Dataset

from lib.util.config import hyper


class VideoDatasetProxy(Dataset):
    def __init__(self, datasets, processing=None):
        super().__init__()

        self.datasets = datasets
        self.processing = processing

        self.weights = hyper.weights if hyper.weights else [len(d) for d in datasets]
        self.max_interval = hyper.max_interval
        self.num_template = hyper.num_template
        self.num_search = hyper.num_search

    def sample_video_id(self, dataset):
        for c in count():
            video_id = random.randrange(len(dataset))
            anno = dataset.get_annotation(video_id)

            valid = (anno.groundtruth[..., 2] > 0) & (anno.groundtruth[..., 3] > 0)
            visible = 1 - (anno.occlusion | anno.out_of_view)
            return video_id, anno

    def sample_frame_ids(self, condition, num=1, start=None, stop=None):
        if start is None or start < 0:
            start = 0
        if stop is None or stop > len(condition):
            stop = len(condition)

        optional_ids = [i for i in range(start, stop) if condition[i]]

        if optional_ids:
            return random.choices(optional_ids, k=num)
        return None

    def transform_optional_ids(self, condition):
        optional_ids = [i for i in range(len(condition)) if condition[i]]
        if len(optional_ids) < self.num_template + self.num_search:
            raise Exception('Not enough optional frame ids for num_template and num_search.')

        return optional_ids

    def causal_manner_sample(self, condition):
        optional_ids = self.transform_optional_ids(condition)

        base_frame = random.randrange(self.num_template - 1, len(optional_ids) - self.num_search)

        template_ids = []
        if self.num_template > 1:
            start = 0
            for i in range(1, base_frame):
                start = i

                if optional_ids[i] + self.max_interval > optional_ids[base_frame]:
                    break

            template_ids = random.choices(optional_ids[start:base_frame], k=self.num_template - 1)

        template_ids.append(optional_ids[base_frame])

        stop = base_frame + 2
        for i in range(base_frame + 2, len(optional_ids)):
            stop = i + 1

            if optional_ids[base_frame] + self.max_interval < optional_ids[i]:
                break

        search_ids = random.choices(optional_ids[base_frame + 1:stop], k=self.num_search)

        return sorted(template_ids), sorted(search_ids)

    def trident_manner_sample(self, condition):
        optional_ids = self.transform_optional_ids(condition)

        reverse = random.random() < 0.5

        if reverse:
            optional_ids = reversed(optional_ids)

        static_template = random.randrange(len(optional_ids) - 2)
        search = random.randrange()