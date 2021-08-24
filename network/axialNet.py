import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import torchvision
import torch.nn.functional as F
import numpy as np
#from IPython.core.debugger import set_trace

from torch.utils import model_zoo
#import deeplab_resnet
from torch.autograd import Variable
import scipy.misc
from PIL import Image
import math
import torch.backends.cudnn as cudnn

class q_transform(nn.Conv2d):
    """Conv2d for q_transform"""

class k_transform(nn.Conv2d):
    """Conv2d for q_transform"""

class v_transform(nn.Conv2d):
    """Conv2d for q_transform"""

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def convrelu(in_channels, out_channels, kernel_size=1, stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class last_upconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = convrelu(in_channels, in_channels//2, kernel_size=3, stride=1,padding=1)
            self.conv2 = convrelu(in_channels//2, out_channels,kernel_size=3, stride=1,padding=1)

        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = convrelu(in_channels, out_channels)

    def forward(self, x):

        x1 = self.up(x)
        x2 = self.conv1(x1)
        x3 = self.up(x2)
        x4 = self.conv2(x3)


        return x4


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,upsample=None,
                 stride=1, bias=False, width=False):
        # assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.upsample = upsample

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
            if self.width:
                x = x.permute(0, 2, 1, 3)
            else:
                x = x.permute(0, 3, 1, 2)  # N, W, C, H
            N, W, C, H = x.shape
            x = x.contiguous().view(N * W, C, H)

            # Transformations
            qkv = self.bn_qkv(self.qkv_transform(x))
            q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

            # Calculate position embedding
            all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
            q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
            qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
            kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
            qk = torch.einsum('bgci, bgcj->bgij', q, k)
            stacked_similarity = torch.cat([qk, qr, kr], dim=1)
            stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
            #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
            # (N, groups, H, H, W)
            similarity = F.softmax(stacked_similarity, dim=3)
            sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
            sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
            stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
            output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

            if self.width:
                output = output.permute(0, 2, 1, 3)
            else:
                output = output.permute(0, 2, 3, 1)

            if self.upsample is not None:
                pass
            else:
                if self.stride > 1:
                    output = self.pooling(output)

            return output

    def reset_parameters(self):
            self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
            #nn.init.uniform_(self.relative, -0.1, 0.1)
            nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

 

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print('x', x.shape)
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#AxialDecodeBlock changes here!!!
class AxialDecodeBlock(nn.Module):
    expansion = 2
           
    def __init__(self, inchannel, outchannel, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):

        super(AxialDecodeBlock, self).__init__()
       
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
   
        self.conv_down = conv1x1(inchannel, outchannel) #be inplanes, width
        # hwere is upsampling here?
        self.bn1 = norm_layer(outchannel)
        
        self.hight_block = AxialAttention(outchannel, outchannel, groups=groups, kernel_size=kernel_size, upsample= None) #be
        self.width_block = AxialAttention(outchannel, outchannel, groups=groups, kernel_size=kernel_size, stride=stride, width=True, upsample = upsample) #be
        self.conv_up = conv1x1(outchannel,outchannel) #be width, planes * self.expansion
        self.bn2 = norm_layer(outchannel) 
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print(x.shape)
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # out = self.upsample(out) # 
        # print('n', out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)
        out = self.conv_up(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.upsample is not None:
          if out.shape[1] != 32:
              out = self.upsample(out)
              out = self.relu(out)

        return out

class AxialAttentionNet(nn.Module):

    def __init__(self, block, decodeblock,layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.5):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=56)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=56,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=28,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=14,
                                       dilate=replace_stride_with_dilation[2])
        self.layer3_1x1 = convrelu(512, 512, kernel_size = 1,stride=1)
        self.layer2_1x1 = convrelu(256, 256, kernel_size = 1,stride=1)
        self.layer1_1x1 = convrelu(128, 128, kernel_size = 1,stride=1)
        self.layer0_1x1 = convrelu(32, 32, kernel_size = 1,stride=1)

        self.layer4_decode = self._decoder_layer(decodeblock, 1024,512, stride = 2,  kernel_size=7)
        self.layer3_decode = self._decoder_layer(decodeblock, 512,256, stride = 2, kernel_size=14)
        self.layer2_decode = self._decoder_layer(decodeblock, 256,128, stride = 2, kernel_size=28)
        self.layer1_decode = self._decoder_layer(decodeblock, 128,32, stride = 1, kernel_size=56)

        self.sideconv4_1x1 = convrelu(1024, 512, kernel_size=1, stride=1)
        self.sideconv3_1x1 = convrelu(512, 256, kernel_size=1,stride=1)
        self.sideconv2_1x1 = convrelu(256, 128, kernel_size = 1, stride = 1)
        self.sideconv1_1x1 = convrelu(128, 32, kernel_size = 1, stride = 1)

        self.dconv4 = convrelu(1024, 512, kernel_size=3, stride=1, padding = 1)
        self.dconv3 = convrelu(512, 256, kernel_size=3,stride=1,padding = 1)
        self.dconv2 = convrelu(256, 128, kernel_size = 3, stride = 1,padding = 1)
        self.dconv1 = convrelu(128, 32, kernel_size = 3, stride = 1,padding = 1)
        self.dconvlast = convrelu(64, 32, kernel_size = 1, stride = 1)

        self.bn4 = norm_layer(512)
        self.bn3 = norm_layer(256)
        self.bn2 = norm_layer(128)
        self.bn1 = norm_layer(32)



        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_last = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_last = last_upconv(in_channels = 32, out_channels = 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, q_transform) or isinstance(m, k_transform) or isinstance(m, v_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _decoder_layer(self, decodeblock, inchannel, outchannel, kernel_size = 56, stride=1, dilate = False):


          upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #be 
          norm_layer = self._norm_layer
          previous_dilation = self.dilation
          decoder_layers = []
          decoder_layers.append(decodeblock(inchannel, outchannel, stride, upsample, groups=self.groups,
                              base_width=self.base_width, dilation=previous_dilation, 
                              norm_layer=norm_layer, kernel_size=kernel_size))
          return nn.Sequential(*decoder_layers)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        layer0 = self.maxpool(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_axial = self.layer4_decode(layer4) #box

        side_4 = self.upsample(self.sideconv4_1x1(layer4)) #side ko 
        layer3_conv1x1 = self.layer3_1x1(layer3) #from encoder

        layer4_sum = torch.cat([layer4_axial +  side_4 , layer3_conv1x1], dim=1)
        layer4_out = self.dconv4(layer4_sum)

        layer3_axial = self.layer3_decode(layer4_out) 
        side_3 = self.upsample(self.sideconv3_1x1(layer4_out)) #side ko
        layer2_conv1x1 = self.layer2_1x1(layer2)
        layer3_sum = torch.cat([layer3_axial +  side_3,  layer2_conv1x1], dim=1)
        layer3_out = self.dconv3(layer3_sum)

        layer2_axial = self.layer2_decode(layer3_out)
        side_2 = self.upsample(self.sideconv2_1x1(layer3_out)) # side ko
        layer1_conv1x1 = self.layer1_1x1(layer1)
        layer2_sum = torch.cat([layer2_axial +  side_2, layer1_conv1x1], dim=1)
        layer2_out = self.dconv2(layer2_sum)

        layer1_axial = self.layer1_decode(layer2_out)  
        side_1 = self.sideconv1_1x1(layer2_out) #side ko
        layer0_conv1x1 = self.layer0_1x1(layer0)
        layer1_sum = torch.cat([layer1_axial + side_1, layer0_conv1x1], dim=1)
        layer1_out = self.dconvlast(layer1_sum)


        convlast = self.conv_last(layer1_out)
        return convlast

    def forward(self, x):
        f = self._forward_impl(x)

        return f

def axial50s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, AxialDecodeBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    #s=0.5
    return model


import math
import os
import torch
import torch.nn.functional as F

def save_model(model, optimizer, epoch, args):
    os.system('mkdir -p {}'.format(args.work_dirs))
    if optimizer is not None:
        torch.save({
            'net': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(args.work_dirs, '{}.pth'.format(epoch)))
    else:
        torch.save({
            'net': model.state_dict(),
            'epoch': epoch
        }, os.path.join(args.work_dirs, '{}.pth'.format(epoch)))



def load_model(network, args):
    if not os.path.exists(args.work_dirs):
        print("No such working directory!")
        raise AssertionError

    pths = [pth.split('.')[0] for pth in os.listdir(args.work_dirs) if 'pth' in pth]
    if len(pths) == 0:
        print("No model to load!")
        raise AssertionError

    pths = [int(pth) for pth in pths]
    if args.test_model == -1:
        pth = -1
        if pth in pths:
            pass
        else:
            pth = max(pths)
    else:
        pth = args.test_model
    try:
        if args.distributed:
            loc = 'cuda:{}'.format(args.gpu)
            model = torch.load(os.path.join(args.work_dirs, '{}.pth'.format(pth)), map_location=loc)
    except:
        model = torch.load(os.path.join(args.work_dirs, '{}.pth'.format(pth)))
    try:
        network.load_state_dict(model['net'], strict=True)
    except:
        network.load_state_dict(convert_model(model['net']), strict=True)
    return True

def convert_model(model):
    new_model = {}
    for k in model.keys():
        new_model[k[7:]] = model[k]
    return new_model

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update_loss(self, val):
        self.sum += val.detach().cpu()
        self.n += 1
    def update_acc(self, val):
        self.sum += val
        self.n += 1
    def get_sum(self):
        return self.sum

    @property
    def avg(self):
        return self.sum / self.n
    
def build_model(args):
    model = axial50s(pretrained=False)
    return model


