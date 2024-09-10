from functools import partial

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from einops import rearrange

from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp
from .utils import get_2d_sincos_pos_embed
import numpy as np

# copy from detectron2
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# copy from detectron2
class Conv_LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MMR_(nn.Module):
    def __init__(self, img_size=224, patch_size=4, mask_ratio=0.05, in_chans=3, decoder_embed_dim=768,
                 norm_pixel_loss=False, depths=(2, 2, 6, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
                 window_size=7, qkv_bias=True, mlp_ratio=4., drop_path_rate=0.1, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=None, patch_norm=True,
                 cfg=None, scale_factors=(4.0, 2.0, 1.0), FPN_output_dim=(256, 512, 1024)):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        #self.norm_pixel_loss = norm_pixel_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  #different

        self.layers = self.build_layers()
        self.first_patch_expanding = PatchExpanding(dim= decoder_embed_dim, norm_layer= norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)

        decoder_embed_dim = embed_dim
        self.decoder_FPN_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_FPN_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                                  requires_grad=False)  # fixed sin-cos embedding

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE Simple FPN specifics
        # for scale = 4, 2, 1
        strides = [int(patch_size / scale) for scale in scale_factors]  # [4, 8, 16]

        self.stages = []
        use_bias = False
        for idx, scale in enumerate(scale_factors):
            out_dim = 1536
            decoder_embed_dim = out_dim
            print(decoder_embed_dim)
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim // 2, kernel_size=2, stride=2),
                    Conv_LayerNorm(decoder_embed_dim // 2),
                    nn.LeakyReLU(),
                    #nn.GELU(),
                    nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 2, kernel_size=2, stride=2),
                    nn.LeakyReLU(),
                    #nn.GELU(),
                    nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 4, kernel_size=2, stride=2)
                ]
                out_dim = decoder_embed_dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim // 2, kernel_size=2, stride=2),
                          nn.LeakyReLU(),
                          #nn.GELU(),
                          nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 2, kernel_size=2, stride=2)
                          ]
                out_dim = decoder_embed_dim // 2
            elif scale == 1.0:
                layers = [nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim, kernel_size=2, stride=2),]
                out_dim = decoder_embed_dim
            elif scale == 0.5:#
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        FPN_output_dim[idx],
                        kernel_size=1,
                        bias=use_bias,
                        norm=Conv_LayerNorm(FPN_output_dim[idx]),
                    ),

                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)
        # --------------------------------------------------------------------------
        self.cfg = cfg

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_FPN_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5), cls_token=True)
        self.decoder_FPN_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder_FPN_mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')
        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(ids_keep, d, rounding_mode='floor') * d * r ** 2 + ids_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :ids_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, ids_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.cls_token
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer)
            layers_up.append(layer)
        return layers_up
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        #print("x in forward_encoder", x.shape)

        x, mask = self.window_masking(x, remove=False, mask_len_sparse=False)
        #print("x", x.shape)
        #print('mask', mask.shape)

        for layer in self.layers:
            x = layer(x)
        return x, mask

    def forward_decoder_FPN(self, x):
        # append mask tokens to sequence
        x_ = x

        # FPN stage
        h = int(x_.shape[1])
        w = int(x_.shape[2])
        decoder_dim = x_.shape[3]

        x_ = x_.permute(0, 3, 1, 2)
        results = []

        for idx, stage in enumerate(self.stages):
            stage_feature_map = stage(x_)
            results.append(stage_feature_map)

        return {layer: feature for layer, feature in zip(self.cfg.TRAIN.MMR.layers_to_extract_from, results)}

    def forward(self, x):
        latent, mask = self.forward_encoder(x)
        #print("latent.shape", latent.shape)
        reverse_features = self.forward_decoder_FPN(latent)  # [N, L, p*p*3]
#        print("reverse_features.shape", reverse_features.shape)
        return reverse_features


def MMR_base(**kwargs):
    model = MMR_(
        img_size=224, patch_size=4, in_chans=3,
        decoder_embed_dim=1536,
        depths=(2, 22, 22, 2), embed_dim=192, num_heads=(32, 32, 32, 32),
        window_size=7, qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.2, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def get_abs_pos(abs_pos, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    new_abs_pos = F.interpolate(
        abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )

    return new_abs_pos.permute(0, 2, 3, 1).reshape(1, int(h * w), -1)
