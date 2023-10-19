import math

import torch
from torchvision.transforms import functional


# ===================
# = Image Operators =
# ===================

def crop_square_from_image_by_box(image, box, search_area_factor, output_size=None):
    """Crop a square from the image by calculating the relative area ratio.

    Args:
        image: The image to be cropped.
        box: The reference box.
        search_area_factor: The ratio of area between the cropped square and the box.
        output_size: Expected square size.
    """
    x, y, w, h = box.tolist()

    square_size = math.ceil(search_area_factor * (w * h) ** 0.5)
    if square_size < 1:
        raise Exception('Invalid square size for image cropping.')

    x1 = round(x + 0.5 * w - 0.5 * square_size)
    y1 = round(y + 0.5 * h - 0.5 * square_size)
    x2 = x1 + square_size
    y2 = y1 + square_size

    _, im_h, im_w = image.shape

    square = functional.pad(image, (-x1, -y1, x2 - im_w, y2 - im_h))
    resize_factor = 1

    if output_size:
        square = functional.resize(square, (output_size, output_size), antialias=True)
        resize_factor = output_size / square_size

    return square, resize_factor


def resize_and_pad(image, size):
    *_, h, w = image.shape
    scale = size / max(h, w)
    h, w = int(h * scale), int(w * scale)

    image = functional.resize(image, (h, w), antialias=True)

    right = size - w
    bottom = size - h

    return functional.pad(image, [0, 0, right, bottom])


# =================
# = Box Operators =
# =================

def transform_box_relative(box_in, box_ref, square_size, resize_factor, normalize=False):
    """Transform the box from absolute coordinates to relative coordinates.

    Args:
        box_in: The input boxes. (x, y, w, h).
        box_ref: The reference boxes. (x, y, w, h).
        square_size: The square size.
        resize_factor: The resize factor of square.
        normalize: Whether to perform normalization.
    """
    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]
    box_ref_center = box_ref[0:2] + 0.5 * box_ref[2:4]

    box_out_center = 0.5 * square_size + resize_factor * (box_in_center - box_ref_center)
    box_out_size = resize_factor * box_in[2:4]

    box_out = torch.cat((box_out_center - 0.5 * box_out_size, box_out_size))

    if normalize:
        box_out = box_out / square_size

    return box_out
