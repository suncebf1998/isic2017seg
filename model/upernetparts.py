# from types import NoneType
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from mmcv.cnn import ConvModule
from abc import ABCMeta, abstractmethod
from mmcv.cnn import normal_init
from functools import partial
ConvModule = partial(ConvModule, inplace=False)
# nn.ReLU

def make_cfg(key:str):
    if isinstance(key, dict) or key is None:
        return key
    return dict(type=key)

class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                #  loss_decode=dict(
                #      type='CrossEntropyLoss',
                #      use_sigmoid=False,
                #      loss_weight=1.0),
                 ignore_index=255,
                #  sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        # self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        # if sampler is not None:
        #     self.sampler = build_pixel_sampler(sampler, context=self)
        # else:
        #     self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    # mode='bilinear',
                    # align_corners=self.align_corners
                    ) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

''' remove
    # @auto_fp16()
    # @abstractmethod
    # def forward(self, inputs):
    #     """Placeholder of forward function."""
    #     pass

    # def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
    #     """Forward function for training.
    #     Args:
    #         inputs (list[Tensor]): List of multi-level img features.
    #         img_metas (list[dict]): List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:Collect`.
    #         gt_semantic_seg (Tensor): Semantic segmentation masks
    #             used if the architecture supports semantic segmentation task.
    #         train_cfg (dict): The training config.

    #     Returns:
    #         dict[str, Tensor]: a dictionary of loss components
    #     """
    #     seg_logits = self.forward(inputs)
    #     losses = self.losses(seg_logits, gt_semantic_seg)
    #     return losses

    # def forward_test(self, inputs, img_metas, test_cfg):
    #     """Forward function for testing.

    #     Args:
    #         inputs (list[Tensor]): List of multi-level img features.
    #         img_metas (list[dict]): List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:Collect`.
    #         test_cfg (dict): The testing config.

    #     Returns:
    #         Tensor: Output segmentation map.
    #     """
    #     return self.forward(inputs)

    # @force_fp32(apply_to=('seg_logit', ))
    # def losses(self, seg_logit, seg_label):
    #     """Compute segmentation loss."""
    #     loss = dict()
    #     seg_logit = resize(
    #         input=seg_logit,
    #         size=seg_label.shape[2:],
    #         mode='bilinear',
    #         align_corners=self.align_corners)
    #     if self.sampler is not None:
    #         seg_weight = self.sampler.sample(seg_logit, seg_label)
    #     else:
    #         seg_weight = None
    #     seg_label = seg_label.squeeze(1)
    #     loss['loss_seg'] = self.loss_decode(
    #         seg_logit,
    #         seg_label,
    #         weight=seg_weight,
    #         ignore_index=self.ignore_index)
    #     loss['acc_seg'] = accuracy(seg_logit, seg_label)
    #     return loss   
'''

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                # mode='bilinear',
                # align_corners=self.align_corners
                )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        """
        Args:
            pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
                    Module.
            **kwargs:
                in_channels (int): Input channels.
                channels (int): Channels after modules, before conv_seg.
                conv_cfg (dict|None): Config of conv layers.
                norm_cfg (dict|None): Config of norm layers.
                act_cfg (dict): Config of activation layers.
                align_corners (bool): align_corners argument of F.interpolate.
                num_classes (int): Number of classes.
                dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
                in_index (int|Sequence[int]): Input feature index. Default: -1
                input_transform (str|None): Transformation type of input features.
                    Options: 'resize_concat', 'multiple_select', None.
                    'resize_concat': Multiple feature maps will be resize to the
                        same size as first one and than concat together.
                        Usually used in FCN head of HRNet.
                    'multiple_select': Multiple feature maps will be bundle into
                        a list and passed into decode head.
                    None: Only one select feature map is allowed.
                    Default: None.
                ignore_index (int | None): The label index to be ignored. When using
                    masked BCE loss, ignore_index should be set to None. Default: 255
        """
        input_transform = kwargs.get("input_transform", "multiple_select")
        for cfg in ("conv_cfg", "norm_cfg", "act_cfg"):
            kwargs[cfg] = make_cfg(kwargs.get(cfg, None))
        super(UPerHead, self).__init__(
            input_transform=input_transform, **kwargs)

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        new_laterals = [None] * used_backbone_levels
        new_laterals[-1] = laterals[-1]
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            new_laterals[i - 1] = laterals[i - 1] + resize(new_laterals[i], size=prev_shape)
            # laterals[i - 1] += resize(
            #     laterals[i],
            #     size=prev_shape,
            #     # mode='bilinear',
            #     # align_corners=self.align_corners
            #     )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                # mode='bilinear',
                # align_corners=self.align_corners
                )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output

        
if __name__ == "__main__":
    def get_parameter_number(model: torch.nn.Module, printable:bool=False) -> dict:
        """
        stat the total param num and the num of trainable
        model: the model to be evaluated.
        ret: the dict of "Total" and "Trainable"
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info = {'Total': total_num, 'Trainable': trainable_num}
        if printable:
            for key, value in model_info.items():
                print(key, value, sep="\t")
        return model_info
    device = "cuda:0"
    
    outputs = []
    in_channels = []
    for i in range(4, 0, -1): # 4, 3, 2, 1
        size = 14 * 2**i
        channel = 6 * 2**i
        in_channels.append(channel)
        output = torch.randn(16, channel, size, size).to(device)
        print(f"output shape: ({channel}, {size}, {size})")
        outputs.append(output)
    uperhead = UPerHead(in_channels=in_channels, in_index=list(range(4)), channels=64, num_classes=1, norm_cfg="BN2d").to(device)
    print(uperhead)
    get_parameter_number(uperhead, True)
    final_output = uperhead(outputs)
    print(final_output.shape)
    print(final_output.dtype)
