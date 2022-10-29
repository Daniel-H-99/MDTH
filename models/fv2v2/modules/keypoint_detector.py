from numpy import place
from torch import embedding_renorm_, nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from modules.util import KPHourglass, MeshEncoder, make_coordinate_grid, AntiAliasInterpolation2d, ResBottleneck, Resnet1DEncoder, LinearEncoder, BiCategoricalEncodingLayer, get_rotation_matrix, headpose_pred_to_degree, ResnetEncoder, AdaIn

class KPDetector(nn.Module):
    """
    Detecting canonical keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, reshape_channel, reshape_depth,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False):
        super(KPDetector, self).__init__()

        self.predictor = KPHourglass(block_expansion, in_features=image_channel,
                                     max_features=max_features,  reshape_features=reshape_channel, reshape_depth=reshape_depth, num_blocks=num_blocks)

        # self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=7, padding=3)
        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=3, padding=1)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            # self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=7, padding=3)
            self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=3, padding=1)
            '''
            initial as:
            [[1 0 0]
             [0 1 0]
             [0 0 1]]
            '''
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).to(heatmap.device)
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 9, final_shape[2],
                                                final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 3, 3)
            out['jacobian'] = jacobian

        return out


class ImageEncoder(nn.Module):
    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features):
        super(ImageEncoder, self).__init__()

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

        # self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        # self.norm5 = BatchNorm2d(2048, affine=True)
        # self.block6 = ResBottleneck(in_features=2048, stride=2)

        # self.block7 = nn.Sequential()
        # for i in range(2):
        #     self.block7.add_module('b7_'+ str(i), ResBottleneck(in_features=2048, stride=1))


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

        # out = self.conv5(out)
        # out = self.norm5(out)
        # out = F.relu(out)
        # out = self.block6(out)

        # out = self.block7(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        return out

# class KPDetectorGeo(nn.Module):
#     """
#     Detecting canonical keypoints. Return keypoint position and jacobian near each keypoint.
#     """

#     def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, reshape_channel, reshape_depth,
#                  num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False):
#         super(KPDetector, self).__init__()

#         self.Encoder = MeshEncoder()
#         # self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=7, padding=3)


#     def gaussian2kp(self, heatmap):
#         """
#         Extract the mean from a heatmap
#         """
#         shape = heatmap.shape
#         heatmap = heatmap.unsqueeze(-1)
#         grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
#         value = (heatmap * grid).sum(dim=(2, 3, 4))
#         kp = {'value': value}

#         return kp

#     def forward(self, x):
#         if self.scale_factor != 1:
#             x = self.down(x)

#         feature_map = self.predictor(x)
#         prediction = self.kp(feature_map)

#         final_shape = prediction.shape
#         heatmap = prediction.view(final_shape[0], final_shape[1], -1)
#         heatmap = F.softmax(heatmap / self.temperature, dim=2)
#         heatmap = heatmap.view(*final_shape)

#         out = self.gaussian2kp(heatmap)

#         if self.jacobian is not None:
#             jacobian_map = self.jacobian(feature_map)
#             jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 9, final_shape[2],
#                                                 final_shape[3], final_shape[4])
#             heatmap = heatmap.unsqueeze(2)

#             jacobian = heatmap * jacobian_map
#             jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
#             jacobian = jacobian.sum(dim=-1)
#             jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 3, 3)
#             out['jacobian'] = jacobian

#         return out

class ExpTransformer(nn.Module):
    """
    Estimating transformed expression of given target face expression to source identity
    """

    def __init__(self, block_expansion, feature_channel, input_dim, num_kp, image_channel, max_features, num_bins=66, num_layer=1, num_heads=32, code_dim=8, latent_dim=256, estimate_jacobian=True, sections=None):
        super(ExpTransformer, self).__init__()
        self.num_heads = num_heads
        self.num_kp = num_kp
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.delta_id_encoder = LinearEncoder(input_dim=3 * num_kp, latent_dim=self.latent_dim, output_dim=self.latent_dim, depth=2)
        self.delta_kp_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_kp * 3),
            nn.Tanh()
        )
        
        self.delta_style_encoder = LinearEncoder(input_dim=3 * self.num_kp, latent_dim=self.latent_dim, output_dim=self.latent_dim // 2, depth=2)
        self.delta_exp_encoder = LinearEncoder(input_dim=2048, latent_dim=self.latent_dim, output_dim=self.num_heads, depth=2)
        self.delta_exp_code_decoder = nn.Linear(self.num_heads, self.latent_dim // 2)
        
        self.delta_heads_pre_scale = nn.Parameter(torch.zeros(self.num_heads, 1).requires_grad_(True))
        self.delta_heads_post_scale = nn.Parameter(torch.zeros(self.num_heads, 1).requires_grad_(True))
        
        self.delta_decoder = LinearEncoder(input_dim=self.latent_dim, latent_dim=self.latent_dim, output_dim=self.num_kp * 3, depth=2)
        
        init.constant_(self.delta_heads_pre_scale, 0)
        init.constant_(self.delta_heads_post_scale, 0)
        # latent_dim = 2048
        
    def encode(self, x, placeholder=['kp', 'exp', 'style']):
        output = {}
        if 'kp' in placeholder:
            id_embedding = self.delta_id_encoder(x['state'])
            output['kp'] = id_embedding
            
        if 'exp' in placeholder:
            exp = self.delta_exp_encoder(x['feat'])
            delta_exp_code = F.tanh(torch.exp(self.delta_heads_pre_scale / 10).unsqueeze(0).squeeze(2) * exp)
            output['delta_exp_code'] = delta_exp_code
        
        if 'style' in placeholder:
            style = self.delta_style_encoder(x['state'])
            output['delta_style_code'] = style
            
        return output

    def decode(self, embedding):
        res = {}
        if 'kp' in embedding:
            res['kp'] = self.delta_kp_decoder(embedding['kp']).view(len(embedding['kp']), -1, 3)
            
        if 'delta_style_code' in embedding and 'delta_exp_code' in embedding:
            x =  self.delta_exp_code_decoder(torch.exp(self.delta_heads_post_scale / 10).unsqueeze(0).squeeze(2) * embedding['delta_exp_code']) # B x num_heads
            style = embedding['delta_style_code'] # B x num_decoding_layer
            x = torch.cat([x, style], dim=1)
            x = self.delta_decoder(x).view(-1, self.num_kp, 3)
            res['delta'] = x
    
        return res

    def forward(self, src, drv, placeholder=['kp', 'exp', 'style']):
        src_embedding = self.encode(src, placeholder=placeholder)
        drv_embedding = self.encode(drv, placeholder=placeholder)

        src_output = self.decode(src_embedding)
        drv_output = self.decode(drv_embedding)
        # drv_output = self.decode({'kp': src_embedding['kp'], 'style': src_embedding['style'], 'exp': drv_embedding['exp']})

        output = {'src_embedding': src_embedding, 'drv_embedding': drv_embedding}
        
        if 'kp' in placeholder:
            output['kp_src'] = src_output['kp']
            output['kp_drv'] = drv_output['kp']
            
        if 'exp' in placeholder and 'style' in placeholder:
            output['delta_src'] = src_output['delta']
            output['delta_drv'] = drv_output['delta']
            
        return  output

class HEEstimator(nn.Module):
    """
    Estimating head pose and expression.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, num_bins=66, estimate_jacobian=True, sections=None, headmodel_sections=None):
        super(HEEstimator, self).__init__()
        num_kp = 15
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

        self.fc_roll = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)
        self.fc_yaw = nn.Linear(2048, num_bins)

        self.fc_t = nn.Linear(2048, 3)

        self.fc_exp = nn.Linear(2048, 3*num_kp)

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

        yaw = self.fc_roll(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_yaw(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        _yaw = headpose_pred_to_degree(yaw)
        _pitch = headpose_pred_to_degree(pitch)
        _roll = headpose_pred_to_degree(roll)


        R = get_rotation_matrix(_yaw, _pitch, _roll)
        
        # t = torch.cat([t[:, [0]], -t[:, [1]], t[:, [2]]], dim=1)
        # t = t[:, [1, 0, 2]]
        
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp, 'R': R, 'out': out}

