import torch
from torch import nn
from functools import partial
from collections import OrderedDict
import math
from DeciWatch.model.weight_init import trunc_normal_, lecun_normal_


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class OriginalPatchEmbed(nn.Module):
    def __init__(self, patch_dim, embed_dim, patch_height, patch_width, num_patchs):
        super().__init__()
        self.patch_h = patch_height
        self.patch_w = patch_width
        self.num_patches = num_patchs
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # (b, c=3, h, w)->(b, n_patches, patch_size*patch_size*c)
        x = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)

        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_channel, embed_dim, patch_height, patch_width, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=(patch_height, patch_width),
                              stride=(patch_height, patch_width))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # BCHW->BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qvk_bias=False, atten_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, print("dim % num_heads should be 0")
        head_dim = dim//num_heads
        self.scale = head_dim ** (-0.5)

        self.qvk = nn.Linear(dim, dim*3, bias=qvk_bias)
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qvk = self.qvk(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, v, k = qvk.unbind(0)  # make torchscript happy (cannot use tensor as tuple)   shape(B, self.num_heads, N, C//self.num_heads)
        atten = (q @ k.transpose(-2, -1))*self.scale
        atten = atten.softmax(dim=-1)
        atten = self.atten_drop(atten)    # shape(B, self.num_heads, N, N)

        # (B, self.num_heads, N, C//self.num_heads) -> (B, N, self.num_heads, C//self.num_heads) -> (B, N, C)
        x = (atten @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)

        self.fc2=nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate, attention_drop, qvk_bias=False, mlp_ratio=4,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attention = Attention(dim, num_heads, qvk_bias=qvk_bias, atten_drop=attention_drop, proj_drop=drop_rate)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)    # feedForward隐藏层维度
        self.mlp = FeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_rate)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channel, num_classes, embed_dim, depth, num_heads,
                 mlp_ratio=4., qvk_bias=False, drop_rate=0., attn_drop_rate=0., representation_size=None,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, distilled=False, weight_init=''):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patchs = (image_height//patch_height)*(image_width//patch_width)
        patch_dim = in_channel*patch_width*patch_height

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 直接采用原图进行embedding，或者采用con2d提取的特征进行embedding
        if embed_layer is None:
            self.patch_embed = OriginalPatchEmbed(patch_dim, embed_dim, patch_height, patch_width, num_patchs)
            # from einops.layers.torch import Rearrange
            # self.patch_embed = nn.Sequential(
            #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            #     nn.Linear(patch_dim, embed_dim)
            # )
        else:
            self.patch_embed = PatchEmbed(in_channel, embed_dim, patch_height, patch_width)

        self.num_tokens = 2 if distilled else 1
        self.cls_token = nn.Parameter(data=torch.rand(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embedding = nn.Parameter(torch.rand(1, num_patchs+self.num_tokens, embed_dim))  # 位置编码, 多一个cls_token的位置编码
        self.pos_drop = nn.Dropout(drop_rate)

        self.transformer = nn.Sequential()
        for i in range(depth):
            self.transformer.add_module(name=str(i),
                module=TransformerBlock(embed_dim, num_heads, drop_rate, attn_drop_rate,
                                        qvk_bias, mlp_ratio, norm_layer=norm_layer, act_layer=act_layer))
        self.norm = norm_layer(embed_dim)

        self.num_features = embed_dim
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embedding, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)  # shape(B, N, C)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)    # shape(B, N+1, C)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)   # shape(B, N+2, C)
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])   # shape(B, C)
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def named_apply(fn, module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

