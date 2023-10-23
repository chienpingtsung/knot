import abc

import torch
from einops import rearrange

from lib.data import func
from lib.util.config import hyper


class BaseProcessing(abc.ABC):
    def __init__(self, joint_trans, template_trans=None, search_trans=None):
        self.joint_trans = joint_trans
        self.template_trans = template_trans
        self.search_trans = search_trans

    @abc.abstractmethod
    def __call__(self, data):
        pass


class Processing(BaseProcessing):
    def __init__(self, joint_trans, template_trans=None, search_trans=None):
        super().__init__(joint_trans, template_trans, search_trans)

        self.t_scale_jitter = hyper.template.scale_jitter
        self.t_center_jitter = hyper.template.center_jitter
        self.t_area_factor = hyper.template.area_factor
        self.t_size = hyper.template.size
        self.s_scale_jitter = hyper.search.scale_jitter
        self.s_center_jitter = hyper.search.center_jitter
        self.s_area_factor = hyper.search.area_factor
        self.s_size = hyper.search.size

    def jitter_boxes(self, boxes, scale_jitter, center_jitter):
        if len(boxes) < 1:
            return rearrange(torch.Tensor([]), '(n d) -> n d', d=4)

        sizes = boxes[..., 2:4]
        centers = boxes[..., 0:2] + 0.5 * sizes

        sizes = sizes * torch.exp(torch.randn_like(sizes) * scale_jitter)

        offsets = torch.sqrt(torch.prod(sizes, dim=1, keepdim=True)) * center_jitter
        centers = centers + offsets * (torch.rand_like(centers) - 0.5)

        return torch.cat((centers - 0.5 * sizes, sizes), dim=1)

    def process(self, frames, boxes, condition, scale_jitter, center_jitter, area_factor, size):
        condition = condition.numpy().astype(bool)
        boxes_jit = boxes.clone()
        boxes_jit[condition] = self.jitter_boxes(boxes_jit[condition], scale_jitter, center_jitter)

        crop_frames = []
        crop_bbox = []
        for frame, box, box_jit, cond in zip(frames, boxes, boxes_jit, condition):
            if cond:
                frame, resize_factor = func.crop_square_from_image_by_box(frame, box_jit, area_factor, size)
            else:
                *_, h, w = frame.shape
                box_jit = torch.Tensor([0, 0, max(h, w), max(h, w)])
                frame, resize_factor = func.resize_and_pad(frame, size)
            crop_frames.append(frame)

            box = func.transform_box_relative(box, box_jit, size, resize_factor, normalize=True)
            crop_bbox.append(box)

        return crop_frames, crop_bbox

    def __call__(self, data):
        if self.joint_trans:
            data = self.joint_trans(data)

        if self.template_trans:
            data.update(self.template_trans({k: v for k, v in data.items() if 'template_' in k}))

        if self.search_trans:
            data.update(self.template_trans({k: v for k, v in data.items() if 'search_' in k}))

        data.template_frames, data.template_bbox = self.process(
            data.template_frames, data.template_bbox, data.template_visible,
            self.t_scale_jitter, self.t_center_jitter, self.t_area_factor, self.t_size)
        data.search_frames, data.search_bbox = self.process(
            data.search_frames, data.search_bbox, data.search_visible,
            self.s_scale_jitter, self.s_center_jitter, self.s_area_factor, self.s_size)

        return data
