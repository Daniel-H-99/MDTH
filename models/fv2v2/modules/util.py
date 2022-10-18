from turtle import forward
from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import models


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat.inverse(), yaw_mat.inverse(), roll_mat.inverse())

    return rot_mat

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()).to(mean.device)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid_2d(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def make_coordinate_grid(spatial_size, type):
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
   
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed

class BiCategoricalEncodingLayer(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(BiCategoricalEncodingLayer, self).__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.layer = nn.Linear(self.in_dim, self.num_heads)

    def forward(self, x):
        # x: B x in_dim
        x = self.layer(x)
        x = F.sigmoid(x)
        x = 2 * x - 1
        return x

class LinearEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = None, output_dim: int = None, depth=0):
        """
        :param latent_dim: size of the latent expression embedding before quantization through Gumbel softmax
        :param n_vertices: number of face mesh vertices
        :param mean: mean position of each vertex
        :param stddev: standard deviation of each vertex position
        :param model_name: name of the model, used to load and save the model
        """
        super(LinearEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        if self.latent_dim is None:
            self.latent_dim = self.input_dim
        if self.output_dim is None:
            self.output_dim = self.latent_dim
            
            
        self.depth = depth
        
        self.layers = torch.nn.ModuleList()
        if self.depth == 0:
            self.layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            self.layers.append(nn.Linear(self.input_dim, self.latent_dim))
        for _ in range(self.depth - 1):
            self.layers.append(nn.Linear(self.latent_dim, self.latent_dim))
        self.layers.append(nn.Linear(self.latent_dim, self.output_dim))

    def forward(self, x):
        """
        :param geom: B x n_vertices x 3 Tensor containing face geometries
        :return: code: B x latent_dim Tensor containing a latent expression code/embedding
        """
        
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), 0.2)

        x = self.layers[-1](x)

        return x
    

class AdaIn(nn.Module):
    def __init__(self, condition_dim, num_output, hidden_dim=128, depth=2):
        super(AdaIn, self).__init__()
        self.condition_dim = condition_dim
        self.depth = depth
        self.num_output = num_output
        self.condition_encoder = LinearEncoder(input_dim=self.condition_dim, hidden_dim=self.hidden_dim, output_dim=self.num_output, depth=self.depth)
    
    def forward(self, x):
        # x: B x condition_dim
        return self.condition_encoder(x) # B x num_output
    
    def normalize(self, x, mean, std):
        # x: B x D
        # x, std: B
        mean_x = x.mean(dim=-1, keepdim=True)
        std_x = x.std(dim=-1, keepdim=True)
        normed_x = (x - mean_x) / std_x.clamp(min=1e-6)
        return mean.unsqueeze(1) + normed_x * std.unsqueeze(1)
    
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class MeshEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, n_vertices: int = 68, num_kp: int = 478, mean=None, stddev=None):
        """
        :param latent_dim: size of the latent expression embedding before quantization through Gumbel softmax
        :param n_vertices: number of face mesh vertices
        :param mean: mean position of each vertex
        :param stddev: standard deviation of each vertex position
        :param model_name: name of the model, used to load and save the model
        """
        super(MeshEncoder, self).__init__()

        self.n_vertices = n_vertices
        self.num_kp = num_kp
        self.latent_dim = latent_dim

        shape = (1, self.n_vertices, 3)
        self.register_buffer("mean", torch.zeros(shape) if mean is None else mean.view(shape))
        self.register_buffer("stddev", torch.ones(shape) if stddev is None else stddev.view(shape))

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(self.n_vertices * 3, 256),
            torch.nn.Linear(256, self.latent_dim),
        ])

        self.code = torch.nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, geom):
        """
        :param geom: B x n_vertices x 3 Tensor containing face geometries
        :return: code: B x latent_dim Tensor containing a latent expression code/embedding
        """
        x = (geom - self.mean) / self.stddev
        x = x.view(x.shape[0], self.n_vertices*3)

        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)

        output = self.code(x)

        return {'latent': x, 'output': output}
    
class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        image_channel = 3
        block_expansion = 64
        self.conv1 = nn.Conv2d(in_channels=image_channel, out_channels=block_expansion, kernel_size=7, padding=3, stride=2)
        self.norm1 = BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=block_expansion, out_channels=256, kernel_size=1)
        self.norm2 = BatchNorm2d(256, affine=True)

        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module('b1_'+ str(i), ResBottleneck(in_features=256, stride=1))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.norm3 = BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)

        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module('b3_'+ str(i), ResBottleneck(in_features=512, stride=1))

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.norm4 = BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)

        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module('b5_'+ str(i), ResBottleneck(in_features=1024, stride=1))

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.norm5 = BatchNorm2d(2048, affine=True)
        self.block6 = ResBottleneck(in_features=2048, stride=2)

        self.block7 = nn.Sequential()
        for i in range(2):
            self.block7.add_module('b7_'+ str(i), ResBottleneck(in_features=2048, stride=1))


    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.block2(out)

        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.block4(out)

        out = self.block5(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = F.relu(out)
        out = self.block6(out)

        out = self.block7(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        return out


class conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if 'alpha' in args['test.tsa_opt']:
            # residual adaptation in matrix form
            y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        return y

class resnet_tsa(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet, num_layer=1):
        super(resnet_tsa, self).__init__()
        self.num_layer = num_layer
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for i in range(self.num_layer):
            for block in getattr(orig_resnet, f'layer{i}'):
                for name, m in block.named_children():
                    if isinstance(m, nn.Conv2d):
                        new_conv = conv_tsa(m)
                        setattr(block, name, new_conv) 

        self.backbone = orig_resnet

    def forward(self, x):
        x = self.backbone(x)

class ResBlock1d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features):
        super(ResBlock1d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1)
        self.norm1 = BatchNorm2d(out_features, affine=True)
        self.norm2 = BatchNorm2d(out_features, affine=True)
        self.in_features = in_features
        self.out_features = out_features
        if self.in_features != self.out_features:
            self.conv3 = nn.Conv2d(in_channels=self.in_features, out_channels=self.out_features, kernel_size=1)
            self.norm3 = BatchNorm2d(out_features, affine=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.in_features != self.out_features:
            x = self.conv3(x)
            x = self.norm3(x)
        out += x
        out = F.relu(out)

        return out

class Resnet1DEncoder(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim):
        super(Resnet1DEncoder, self).__init__()
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        for i in range(num_layer):
            setattr(self, f'layer{i}', ResBlock1d(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim))

    def forward(self, x):
        # x: B x d
        x = x.unsqueeze(2).unsqueeze(3)
        for i in range(self.num_layer):
            x = getattr(self, f'layer{i}')(x)

        x = x.flatten(1)

        return x

class ResBottleneck(nn.Module):
    def __init__(self, in_features, stride):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features, kernel_size=1)
        self.norm1 = BatchNorm2d(in_features//4, affine=True)
        self.norm2 = BatchNorm2d(in_features//4, affine=True)
        self.norm3 = BatchNorm2d(in_features, affine=True)

        self.stride = stride
        if self.stride != 1:
            self.skip = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=stride)
            self.norm4 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.stride != 1:
            x = self.skip(x)
            x = self.norm4(x)
        out += x
        out = F.leaky_relu(out)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        # out = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=(1, 2, 2))
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        # self.out_filters = block_expansion
        self.out_filters = block_expansion + in_features

        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        # for up_block in self.up_blocks[:-1]:
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        # out = self.up_blocks[-1](out)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class KPHourglass(nn.Module):
    """
    Hourglass architecture.
    """ 

    def __init__(self, block_expansion, in_features, reshape_features, reshape_depth, num_blocks=3, max_features=256):
        super(KPHourglass, self).__init__()
        
        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.down_blocks.add_module('down'+ str(i), DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                                                   min(max_features, block_expansion * (2 ** (i + 1))),
                                                                   kernel_size=3, padding=1))

        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=reshape_features, kernel_size=1)

        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_filters = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module('up'+ str(i), UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def forward(self, x):
        out = self.down_blocks(x)
        out = self.conv(out)
        bs, c, h, w = out.shape
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        out = self.up_blocks(out)

        return out
        


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)