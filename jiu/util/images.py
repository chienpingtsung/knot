import torch
import torchvision


def draw_bounding_boxes(image, boxes, width, in_fmt):
    """

    Args:
        image: Tensor image ranges in (0, 1) with shape (C, H, W).
        boxes: Boxes with shape (N, 4).
        width:
        in_fmt: Boxes format.

    Returns:
        image: Tensor image ranges in (0, 255) with shape (C, H, W).
    """
    image = (image * 255).to(torch.uint8)
    if in_fmt != 'xyxy':
        boxes = torchvision.ops.box_convert(boxes, in_fmt, 'xyxy')
    return torchvision.utils.draw_bounding_boxes(image, boxes, width=width)


def batch_draw_bounding_boxes(images, boxes, width, in_fmt):
    """

    Args:
        images: Tensor images ranges in (0, 1) with shape (B, C, H, W).
        boxes: Boxes with shape (B, N, 4).
        width:
        in_fmt: Boxes format.

    Returns:
        images: Tensor images ranges in (0, 255) with shape (B, C, H, W).
    """
    images = list(draw_bounding_boxes(image, box, width, in_fmt) for image, box in zip(images, boxes))
    return torch.stack(images)
