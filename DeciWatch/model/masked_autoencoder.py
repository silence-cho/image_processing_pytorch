import torch
from torch import nn
import torch.nn.functional as F
from model.vision_transformer import TransformerBlock
from DeciWatch.model.weight_init import trunc_normal_, lecun_normal_


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MAE(nn.Module):
    def __init__(self, image_size, patch_size, in_channel, encoder_dim, encoder_depth, encoder_heads,
                 decoder_dim, drop_rate, masking_ratio=0.75, decoder_depth=1, decoder_heads=8):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        image_height, image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)

        # patch embedding
        self.pos_embedding = nn.Parameter(torch.rand(1, self.num_patches, encoder_dim))  # 位置编码, 和vit有区别
        self.pos_drop = nn.Dropout(drop_rate)
        patch_dim = in_channel * self.patch_width * self.patch_height
        self.patch_embedding = nn.Linear(patch_dim, encoder_dim)

        # encoder 部分
        self.encoder = nn.Sequential()
        for i in range(encoder_depth):
            self.encoder.add_module(name=str(i),
                module=TransformerBlock(encoder_dim, encoder_heads, drop_rate=0, attention_drop=0,
                                       mlp_ratio=4, norm_layer=nn.LayerNorm, act_layer=nn.GELU))
        self.norm = nn.LayerNorm(encoder_dim)

        # decoder 部分
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))  # 1个可学习的共享向量
        self.decoder_pos_emb = nn.Embedding(self.num_patches, decoder_dim)  # decoder部分的位置编码？
        self.decoder = nn.Sequential()
        for i in range(decoder_depth):
            self.decoder.add_module(name=str(i),
                module=TransformerBlock(decoder_dim, decoder_heads, drop_rate=0, attention_drop=0,
                                        mlp_ratio=4, norm_layer=nn.LayerNorm, act_layer=nn.GELU))
        # pixel_values_perpatch = patch_dim
        self.to_pixels = nn.Linear(decoder_dim, patch_dim)  # patch_dim: 最后每个patch预测的维度

        self.init_weights()

    def init_weights(self):
        # nn.Embedding 如果不初始化，会默认初始化为标准正态分布
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        device = x.device
        b, c, h, w = x.shape

        # 1. 图片切割为patches
        # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
        patches = x.view(
            b, c,
            h // self.patch_height, self.patch_height,
            w // self.patch_width, self.patch_width
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)

        # 2编码. patch_embedding + pos_embedding
        tokens = self.patch_embedding(patches)
        tokens = tokens + self.pos_embedding

        # 3. 计算mask的数量，获取随机的masked_indices, unmasked_indices
        num_masked = int(self.masking_ratio*self.num_patches)
        # shape(batch, num_patches)
        rand_indices = torch.randn(b, self.num_patches, device=device).argsort(dim=-1)  # argsort返回index
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # 获取unmasked_patch的token，输入encoder
        batch_range = torch.arange(b, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # 获取masked的patch，方便计算重建后的loss
        masked_patches = patches[batch_range, masked_indices]

        # 4. unmask部分经过encoder编码后，加上decoder部分的位置编码
        encoeded_tokens = self.encoder(tokens)
        unmasked_tokens = self.enc_to_dec(encoeded_tokens)
        unmasked_tokens = unmasked_tokens + self.decoder_pos_emb(unmasked_indices)

        # 5. mask部分为一个共享的可学习参数， 加上decoder部分的位置编码
        # shape(, , decoder_dim) -> shape(b, num_masked, decoder_dim)
        mask_tokens = self.mask_token[None, None, :].repeat(b, num_masked, 1)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # 6. mask和unmask两部分进行concat，然后送入decoder
        decoder_tokens = torch.cat((mask_tokens, unmasked_tokens), dim=1)
        decoder_tokens = self.decoder(decoder_tokens)

        # 7. 只取decoder_tokens中mask的部分，每个patch预测patch的所有像素值
        mask_tokens = decoder_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # 8. 只计算mask部分的像素重建loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss


def _init_vit_weights(module: nn.Module, name: str = ''):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)