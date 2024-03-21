from torch import nn

from lib.model.component import CornerPredictor
from lib.model.vit import VisionTransformer


class URT(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.vit = VisionTransformer(nlp_max_length=model_args.nlp_max_length,
                                     img_size_t=model_args.img_size_t,
                                     img_size_s=model_args.img_size_s,
                                     patch_size=model_args.patch_size,
                                     embed_dim=model_args.embed_dim)
        self.corner_predictor = CornerPredictor(in_ch=model_args.embed_dim,
                                                patch_num=model_args.img_size_s // model_args.patch_size,
                                                patch_size=model_args.patch_size,
                                                freeze_bn=model_args.freeze_bn)
        self.confidence = nn.Linear(model_args.embed_dim, 1)

    def forward(self, x):
        nlp_x, x_t, x_s = self.vit(x)
        corner = self.corner_predictor(x_s)
        confidence = self.confidence(nlp_x[:, 0, ...])
        return corner, confidence
