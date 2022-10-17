from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from modules.util import KPHourglass, MeshEncoder, make_coordinate_grid, AntiAliasInterpolation2d, ResBottleneck, Resnet1DEncoder, LinearEncoder, BiCategoricalEncodingLayer, get_rotation_matrix, headpose_pred_to_degree, ResnetEncoder

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
        self.code_dim = code_dim
        self.num_kp = num_kp
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.exp_encoder = nn.Sequential(
            ResnetEncoder(),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, self.latent_dim // 2)
        )
        
        self.id_encoder = MeshEncoder(num_kp=self.num_kp, latent_dim=latent_dim)
        self.kp_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_kp * 3),
            nn.Tanh()
        )
        
        self.style_decoder = nn.Linear(self.latent_dim, self.latent_dim // 2)

        self.vq_exp = BiCategoricalEncodingLayer(self.latent_dim // 2, self.num_heads)
        self.codebook = nn.Parameter(torch.zeros(self.num_heads, self.code_dim).requires_grad_(True))
        self.codebook_pre_scale = nn.Parameter(torch.zeros(self.num_heads, 1).requires_grad_(True))
        self.codebook_post_scale = nn.Parameter(torch.zeros(self.num_heads, 1).requires_grad_(True))
        self.fuser = nn.Sequential(
            nn.Linear(self.latent_dim // 2 + self.num_heads * self.code_dim, self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
 
        self.exp_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 3*num_kp),
        )

        init.kaiming_uniform_(self.codebook)
        init.constant_(self.codebook_pre_scale, 1)
        init.constant_(self.codebook_post_scale, 1)
        # latent_dim = 2048

    # def split_embedding(self, img_embedding):
    #     style_embedding, exp_embedding = img_embedding.split([self.latent_dim // 2, self.latent_dim // 2], dim=1)
    #     exp_code = F.tanh(torch.einsum('bk,kp->bkp', self.vq_exp(exp_embedding), self.codebook_pre_scale).squeeze(2))  # B x num_heads
    #     exp_embedding = self.decode_exp_code(exp_code)

    #     return {'style': style_embedding, 'exp': exp_embedding, 'exp_code': exp_code}

    def decode_exp_code(self, exp_code):
        # exp_code: B x num_heads: [-1, 1] codesW
        exp_embedding = torch.einsum('bk,kp->bkp', exp_code, self.codebook_post_scale * F.normalize(self.codebook))
        exp_embedding = exp_embedding.flatten(1)
        return exp_embedding

    def fuse(self, style, exp):
        input = torch.cat([style, exp], dim=1)
        output = self.fuser(input)
        return output

    def encode(self, x):
        exp_embedding = self.exp_encoder(x['img'])
        id_embedding = self.id_encoder(x['mesh'])
        
        exp_code = F.tanh(torch.einsum('bk,kp->bkp', self.vq_exp(exp_embedding), self.codebook_pre_scale).squeeze(2))  # B x num_heads
        exp_embedding = self.decode_exp_code(exp_code)

        kp = id_embedding
        style = F.normalize(self.style_decoder(id_embedding), dim=-1)

        return {'kp': kp, 'style': style, 'exp': exp_embedding, 'exp_code': exp_code}

    # def kp_encode(self, x):
    #     embedding = F.leaky_relu(self.kp_encoder(x), 0.2)
    #     return embedding

    # def kp_decode(self, x):
    #     return self.kp_decoder(x).view(-1, self.num_kp, 3)
        
    def decode(self, embedding):
        res = {}
        if 'kp' in embedding:
            res['kp'] = 2 * self.kp_decoder(embedding['kp']).view(len(embedding['kp']), -1, 3)
            res['kp'][:, :, 2] = res['kp'][:, :, 2] - 0.33
        if 'style' in embedding and 'exp' in embedding:
            res['exp'] = self.exp_decoder(self.fuse(embedding['style'], embedding['exp'])).view(len(embedding['style']), -1, 3)
            # random_flag = torch.rand(res['exp'].shape).to(res['exp'].device) >= 0.5
            # noise = 0.1 * torch.rand(res['exp'].shape).to(res['exp'].device) * random_flag
            # res['exp'] = res['exp'] + noise
        return res

    def forward(self, src, drv):
        src_embedding = self.encode(src)
        drv_embedding = self.encode(drv)

        src_output = self.decode(src_embedding)
        drv_output = self.decode(drv_embedding)

        return {'src_exp': src_output['exp'], 'drv_exp': drv_output['exp'], 'src_embedding': src_embedding, 'drv_embedding': drv_embedding, 'src_kp': src_output['kp'], 'drv_kp': drv_output['kp']}

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

