import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=5,
                 drop=0., act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, in_dim, eps=1e-6),
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            act_layer(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad,
                      padding_mode='reflect', groups=hidden_dim, bias=False),
            act_layer(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6)
        )

        self.drop = nn.Dropout2d(drop, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, C ,H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H//2, W//2, C).permute(0, 3, 1, 2).contiguous()

        return x



class Attention(nn.Module):
    def __init__(self, dim, head_dim, grid_size=1, drop=0., grid_atten=True):
        super().__init__()
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.drop = nn.Dropout2d(drop, inplace=True)

        self.grid_atten = grid_atten

        if grid_atten:
            self.grid_norm = nn.GroupNorm(1, dim, eps=1e-6)
            self.patchmerging_pool = PatchMerging(dim)
            self.ds_norm = nn.GroupNorm(1, dim, eps=1e-6)
            self.q = nn.Conv2d(dim, dim, 1)
            self.kv = nn.Conv2d(dim, dim * 2, 1)

        else:
            self.grid_norm = nn.GroupNorm(1, dim, eps=1e-6)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.ca_mlp = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim // 4, dim, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = (self.grid_size - H % self.grid_size) % self.grid_size
        pad_w = (self.grid_size - W % self.grid_size) % self.grid_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, H_new, W_new = x.shape
        grid_h, grid_w = H_new // self.grid_size, W_new // self.grid_size

        qkv = self.qkv(self.norm(x))

        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, grid_h,
                          self.grid_size, grid_w, self.grid_size)
        qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3)
        qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        grid_x = (attn @ v)

        grid_x = grid_x.reshape(B, self.num_heads, grid_h, grid_w,
                                self.grid_size, self.grid_size, self.head_dim)
        grid_x = grid_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, C, H, W)
        grid_x = self.grid_norm(x + grid_x)

        if self.grid_atten:
            q = self.q(grid_x).reshape(B, self.num_heads, self.head_dim, -1)
            q = q.transpose(-2, -1)
            kv = self.kv(self.ds_norm(self.patchmerging_pool(grid_x)))
            kv = kv.reshape(B, 2, self.num_heads, self.head_dim, -1)
            kv = kv.permute(1, 0, 2, 4, 3)
            k, v = kv[0], kv[1]

            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            global_x = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
            global_x = global_x + grid_x
        else:
            ca_weight = self.ca_mlp(self.gap(grid_x))
            global_x = grid_x * ca_weight

        if pad_h > 0 or pad_w > 0:
            global_x = global_x[:, :, :H, :W]

        x = self.drop(self.proj(global_x))

        return x


class HAT_NetBlock(nn.Module):
    def __init__(self, dim, num_heads, grid_size=1, mlp_ratio=2, grid_atten=True,
                 drop=0., drop_path=0., act_layer=nn.SiLU):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if grid_atten==True:
            self.attn = Attention(dim, dim // num_heads, grid_size=grid_size, drop=drop)
        else:
            self.attn = Attention(dim, dim // num_heads, grid_size=1, drop=drop)

        self.MLP = InvertedResidual(dim, hidden_dim=dim * mlp_ratio, out_dim=dim,
                                     drop=drop, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.MLP(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, grid_size=8,
                 drop_path=0., use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            HAT_NetBlock(dim=dim,
                         num_heads=num_heads,
                         grid_size=grid_size,
                         grid_atten=True if (i % 2 == 0) else False,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class RHTB(nn.Module):
    def __init__(self, dim, depth, num_heads, grid_size=1,
                 drop_path=0., use_checkpoint=False,
                 resi_connection='1conv'):
        super(RHTB, self).__init__()

        self.dim = dim

        self.residual_group = BasicLayer(dim=dim,
                                         depth=depth,
                                         num_heads=num_heads,
                                         grid_size=grid_size,
                                         drop_path=drop_path,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        out = self.residual_group(x)
        out = self.conv(out)
        return out + x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class HATNetIR(nn.Module):
    def __init__(self, in_chans=3, grid_sizes=[2, 2, 3, 3, 4, 4],
                 embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 mlp_ratio=2., drop_path_rate=0.1,
                 norm_layer=nn.GroupNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(HATNetIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.grid_sizes = grid_sizes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB(dim=embed_dim,
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         grid_size=grid_sizes[i_layer],
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         use_checkpoint=use_checkpoint,
                         resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(1, self.num_features, eps=1e-6)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i, layer in enumerate(self.layers):
            grid_size = self.grid_sizes[i]
            _, _, h, w = x.size()

            mod_pad_h = (grid_size - h % grid_size) % grid_size
            mod_pad_w = (grid_size - w % grid_size) % grid_size

            if mod_pad_h > 0 or mod_pad_w > 0:
                x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            x = layer(x)

            if mod_pad_h > 0 or mod_pad_w > 0:
                x = x[:, :, :h, :w]

        x = self.norm(x)
        return x


    def forward(self, x):
        H, W = x.shape[2:]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)

            x = self.conv_after_body(self.forward_features(x)) + x

            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)

        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]


if __name__ == '__main__':
    upscale = 2
    patch_size = 48

    model = HATNetIR(upscale=upscale, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')

    print(f"Input Patch Size: {patch_size}x{patch_size}")

    x = torch.randn((1, 3, patch_size, patch_size))

    output = model(x)
    print(f"Output SR Image Size: {output.shape}")