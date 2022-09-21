from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, kp2gaussian

from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

class DenseMotionNetworkGeo(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, sections, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False, headmodel=None):
        super(DenseMotionNetworkGeo, self).__init__()
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(feature_channel+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=((num_kp+1)*(compress+1) + 1), max_features=max_features, num_blocks=num_blocks)
        
        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.sections = sections

        self.split_ids = [sec[1] for sec in self.sections]
        
        self.ids_in_sections = []
        for sec in self.sections:
            self.ids_in_sections.extend(sec[0])
        
        # self.kp_extractor = nn.Sequential(
        #     nn.Linear(3 * len(self.ids_in_sections), 256),
        #     # nn.LayerNorm([256]),
        #     nn.Dropout(p=0.3),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     # nn.LayerNorm([256]),
        #     nn.Dropout(p=0.3),
        #     nn.ReLU(),
        #     nn.Linear(256, 3 * self.num_kp),
        #     nn.Tanh()
        # )
        
        self.prior_extractors = nn.ModuleList()
        for i, sec in enumerate(self.sections):
            self.prior_extractors.append(nn.Sequential(
                nn.Linear(3 * len(sec[0]), 256),
                # nn.LayerNorm([256]),
                # nn.Dropout(p=0.3),
                nn.ReLU(),
                nn.Linear(256, 3 * sec[1]),
                nn.Tanh()
            ))

        # headmodel_sections = []
        # for k, v in headmodel.items():
        #     if k == 'sections':
        #         continue
        #     if 'mu_x' in k:
        #         headmodel_sections.append(v)
        #         # print(f'v shape {v.shape}')
        # headmodel_sections = torch.cat(headmodel_sections, dim=0).cpu()
        
        # for i, sec in enumerate(self.sections):
        #     if len(headmodel_sections) < 3 * len(sec[0]):
        #         headmodel_sections = torch.zeros(3 * len(sec[0])).to(headmodel_sections.device)
        #     headmodel_section = torch.zeros_like(headmodel_sections[:3 * len(sec[0])])
        #     headmodel_sections = headmodel_sections[3 * len(sec[0]):]
        #     self.register_buffer(f'headmodel_mu_x_{i}', headmodel_section)
            
            
    def extract_prior(self, kp, use_intermediate=False):
        mesh = kp['value'] # B x N x 3
        # print(f'mesh dev: {mesh.device}')
        # print(f'mesh shape: {mesh.shape}')
        # print(f'mesh type: {mesh.type()}')
        bs = len(mesh)
        
        priors = []
        
        secs = self.split_section(mesh)
        
        for i, sec in enumerate(secs):
            # sec = sec.flatten(1) - getattr(self, f'headmodel_mu_x_{i}')[None]
            sec = sec.flatten(1)

            # print(sec.type())
            prior = self.prior_extractors[i](sec).view(bs, -1, 3) # B x num_prior x 3
            priors.append(prior)
            
        priors = torch.cat(priors, dim=1)
            
        return priors
    
    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[:, sec[0]])
        return res
    
        
    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        device = feature.device

        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['kp'].type()).to(device)

        identity_grid = identity_grid.view(1, 1, d, h, w, 3).to(device)
        coordinate_grid = identity_grid - kp_driving['kp'].view(bs, self.num_kp, 1, 1, 1, 3)
    
        k = coordinate_grid.shape[1]
    
        driving_to_source = coordinate_grid + kp_source['kp'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed


    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        device = feature.device
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).to(device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap
        
    def keypoint_transformation(self, kp, he):
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        t = he['t']
        
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)

        rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
        
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

        t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
        kp_t = kp_rotated + t


        kp_neutralized = kp_t

        kp_transformed = kp_t


        jacobian_transformed = None

        return {'value': kp_transformed, 'jacobian': jacobian_transformed, 'neutralized': {'value': kp_neutralized, 'jacobian': jacobian_transformed}}
        
    def extract_rotation_keypoints(self, kp_source, kp_driving):
        device = kp_source['value'].device
        coords_src = self.extract_prior(kp_source)
        coords_drv = self.extract_prior(kp_driving)
        
        src_normed = coords_src
        drv_normed = coords_drv

        if 'exp' in kp_source and 'exp' in kp_driving:
            drv_normed = src_normed - kp_source['exp'] + kp_driving['exp']

        tmp = torch.cat([src_normed, torch.ones(src_normed.shape[0], src_normed.shape[1], 1).to(device) / kp_source['scale'].unsqueeze(1).unsqueeze(2)], dim=2) # B x N x 4
        tmp = tmp.matmul(kp_source['U']) # B x N x 4
        tmp = tmp[:, :, :3] + torch.tensor([-1, -1, 0]).unsqueeze(0).unsqueeze(1).to(device)
        coords_src = tmp # B x N x 3

        tmp = torch.cat([drv_normed, torch.ones(drv_normed.shape[0], drv_normed.shape[1], 1).to(device) / kp_driving['scale'].unsqueeze(1).unsqueeze(2)], dim=2) # B x N x 4
        tmp = tmp.matmul(kp_driving['U']) # B x N x 4
        tmp = tmp[:, :, :3] + torch.tensor([-1, -1, 0]).unsqueeze(0).unsqueeze(1).to(device)
        coords_drv = tmp # B x N x 3

        return {'src': coords_src, 'drv': coords_drv, 'src_normed': src_normed, 'drv_normed': drv_normed}         
        

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        

        rotation_kps = self.extract_rotation_keypoints(kp_source, kp_driving)

        kp_source['kp'] = rotation_kps['src']
        kp_driving['kp'] = rotation_kps['drv']
        
        kp_source['prior'] = rotation_kps['src_normed']
        kp_driving['prior'] = rotation_kps['drv_normed']
    
        out_dict['kp_source'] = {'value': kp_source['kp']}
        out_dict['kp_driving'] = {'value': kp_driving['kp']}
        
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)
        


        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)
        out_dict['heatmap'] = heatmap

        
        input = torch.cat([heatmap, deformed_feature], dim=2)
        input = input.view(bs, -1, d, h, w)

        mesh_img = torch.zeros(input.shape[0], 1, input.shape[2], input.shape[3], input.shape[4]).to(input.device)
        input = torch.cat([mesh_img, input], dim=1)


        prediction = self.hourglass(input)

        mask = self.mask(prediction)

        mask = F.softmax(mask, dim=1)

        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(feature_channel+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp


    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)
        
        k = coordinate_grid.shape[1]
        
        # if 'jacobian' in kp_driving:
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
        '''
        if 'rot' in kp_driving:
            rot_s = kp_source['rot']
            rot_d = kp_driving['rot']
            rot = torch.einsum('bij, bjk->bki', rot_s, torch.inverse(rot_d))
            rot = rot.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            rot = rot.repeat(1, k, d, h, w, 1, 1)
            # print(rot.shape)
            coordinate_grid = torch.matmul(rot, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
            # print(coordinate_grid.shape)
        '''
        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        input = torch.cat([heatmap, deformed_feature], dim=2)
        input = input.view(bs, -1, d, h, w)

        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
