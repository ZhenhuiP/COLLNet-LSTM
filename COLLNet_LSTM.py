import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
import sys
import os
from ODConv import ODConv2d

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result
def conv_bn1(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', ODConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.LeakyReLU())
    return result
def conv_bn_relu1(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.LeakyReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class COLLNetBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()

        self.pw1 = conv_bn_relu1(in_channels, dw_channels, 1, 1, 0, groups=1)
        # self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn1(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.LeakyReLU(inplace=True)

        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', self.drop_path)
    #
    def forward(self, x):
        # self.pw1 = conv_bn_relu1(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1).to(self.device)
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)

    # def forward(self, x):
    #     out = self.pw1(x)
    #     out = self.large_kernel(out)
    #     out = self.prelkb_bn(out)
    #     out = self.lk_nonlinear(out)
    #     out = self.pw2(out)
    #     return x + self.drop_path(out)


class COLLNetStage(nn.Module):

    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False,      # train with torch.utils.checkpoint to save memory
                 small_kernel_merged=False,
                 norm_intermediate_features=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []

        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = COLLNetBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            self.relu = nn.LeakyReLU(inplace=True)
            blks.append(replk_block)
            blks.append(self.relu)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = get_bn(channels)    #   Only use this with RepLKNet-XL on downstream tasks
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
            else:
                x = blk(x)
        return x

class COLLNet(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 use_checkpoint=False,
                 small_kernel_merged=False,
                 use_sync_bn=False,
                 norm_intermediate_features=False       # for RepLKNet-XL on COCO and ADE20K, use an extra BN to normalize the intermediate feature maps then feed them into the heads
                 ):
        super().__init__()

        if num_classes is None and out_indices is None:
            raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and out_indices is not None:
            raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
        if use_sync_bn:
            enable_sync_bn()

        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers) # 4个
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = ODConv2d(256, 512, kernel_size=3, stride=2, padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(self.inplanes)
        self.norm2 = nn.BatchNorm2d(512)
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):   #四个循环
            layer = COLLNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        if num_classes is not None:
            self.norm = get_bn(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], num_classes)



    def forward_features(self, x):

        if self.out_indices is None:
            #print(f"Input shape before stage: {x.shape}")
            #   Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                #print(f"Shape after stage {stage_idx}: {x.shape}")
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
                    #print(f"Shape after transition {stage_idx}: {x.shape}")
            return x
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    outs.append(self.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return outs

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.forward_features(x)
        if self.out_indices:
            return x
        else:

            x = self.norm(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x


    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


def create_COLLNet_Base(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[2, 3, 2], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet222(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[2, 2, 2], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet223(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[2, 2, 3], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet322(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[3, 2, 2], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet233(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[2, 3, 3], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet332(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[3, 3, 2], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet323(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[3, 2, 3], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet333(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[13, 13, 13], layers=[3, 3, 3], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)


def create_COLLNet31L(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[192,384,768,1536],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNetXL(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[27,27,27,13], layers=[2,2,18,2], channels=[256,512,1024,2048],
                    drop_path_rate=drop_path_rate, small_kernel=None, dw_ratio=1.5,
                    num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_COLLNet_3size_kernal(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return COLLNet(large_kernel_sizes=[5, 5, 5], layers=[2, 3, 2], channels=[64, 128, 256],                    #[31, 27, 19, 13], layers=[2, 3, 2, 2], channels=[64, 128, 256, 512],
                    drop_path_rate=drop_path_rate, small_kernel=3, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

class COLLNet_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super().__init__()
        self.n_layer = n_layer
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.spatial =create_COLLNet_Base(num_classes=n_class, use_checkpoint=False)

        self.spatial.head = nn.Sequential()
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=hidden_dim, num_layers=n_layer,  batch_first=True)
        self.liner = nn.Linear(hidden_dim, n_class)

    def structural_reparam(self):
        self.spatial.structural_reparam()

    def forward(self, x):
        batch_size, time_steps = x.shape[0], x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        spatial_features = self.spatial(x)
        spatial_features = spatial_features.view(batch_size, time_steps, -1)
        out, _ = self.lstm(spatial_features)
        out = self.liner(out)
        return out


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    time_steps = 8

    model = COLLNet_LSTM(in_dim=512, hidden_dim=128, n_layer=1, n_class=4)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    model.to(device)
    model.eval()
    print('------------------- training-time model -------------')
    print(model)

    x = torch.randn(2, time_steps, 3, 224, 224)
    x = x.to(device)
    origin_y = model(x)

    model.structural_reparam()
    print('------------------- after re-param -------------')
    print(model)

    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters after reparam: {total_params_after:,}')
    print(f'Trainable parameters after reparam: {trainable_params_after:,}')

    reparam_y = model(x)
    print('------------------- the difference is ------------------------')
    print((origin_y - reparam_y).abs().sum())