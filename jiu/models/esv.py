import einops
import torch
from torch import nn
from transformers import BertModel

from jiu.models import vit
from jiu.models.swin import SwinTransformer
from jiu.util.arguments import hyper


class EventSensitiveVision(nn.Module):
    def __init__(self, reg_token=True, block_fn=vit.Block):
        super().__init__()

        self.lang_model = BertModel.from_pretrained(hyper.lang_model)
        self.vision_model = SwinTransformer()
        self.vision_model.norm = nn.Identity()
        self.vision_model.head = nn.Identity()

        self.reg_token = nn.Parameter(torch.zeros(1, 1, hyper.embed_dim)) if reg_token else None
        self.cmt = nn.Sequential(*[block_fn(hyper.embed_dim, norm_first=False) for i in range(hyper.depth)])
        self.reg_head = nn.Linear(768, 4)
        self.cfs_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, template_img, search_img):
        lang_token, _ = self.lang_model(input_ids, attention_mask, return_dict=False)
        template_token = self.vision_model(template_img)
        template_token = einops.rearrange(template_token, 'b h w c -> b (h w) c')
        search_token = self.vision_model(search_img)
        search_token = einops.rearrange(search_token, 'b h w c -> b (h w) c')

        x = [lang_token, template_token, search_token]
        if self.reg_token:
            x = [self.reg_token] + x
        x = torch.cat(x, dim=1)
        x = self.cmt(x)

        reg = self.reg_head(x[:, 0, :]).squeeze()
        cfs = self.cfs_head(x[:, 1, :]).squeeze()

        return reg, cfs
