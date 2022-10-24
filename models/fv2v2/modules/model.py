from logging import PlaceHolder
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid_2d
from torchvision import models
import numpy as np
from torch.autograd import grad
import modules.hopenet as hopenet
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

'''
# beta version
def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

    return rot_mat
'''

def keypoint_transformation(kp_canonical, mesh):
    device = kp_canonical['value'].device
    
    if 'delta' in mesh:
        kp_normed = kp_canonical['value'] + mesh['delta']
    else:
        kp_normed = kp_canonical['value']
        
    tmp = torch.cat([kp_normed, torch.ones(kp_normed.shape[0], kp_normed.shape[1], 1).to(device) / mesh['scale'].unsqueeze(1).unsqueeze(2)], dim=2) # B x N x 4
    tmp = tmp.matmul(mesh['U']) # B x N x 4
    tmp = tmp[:, :, :3] + torch.tensor([-1, -1, 0]).unsqueeze(0).unsqueeze(1).to(device)
    # tmp[:, :, 2] = -tmp[:, :, 2]
    kp_transformed = tmp # B x N x 3
    
    
    tmp = kp_canonical['value']

    tmp = torch.cat([tmp, torch.ones(kp_normed.shape[0], kp_normed.shape[1], 1).to(device) / mesh['scale'].unsqueeze(1).unsqueeze(2)], dim=2) # B x N x 4
    tmp = tmp.matmul(mesh['U']) # B x N x 4
    tmp = tmp[:, :, :3] + torch.tensor([-1, -1, 0]).unsqueeze(0).unsqueeze(1).to(device)
    kp_canonical_transformed = tmp # B x N x 3

    return {'value': kp_transformed, 'normed': kp_normed, 'canonical': kp_canonical_transformed}         
    
        
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

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat


# def keypoint_transformation(kp_canonical, he, estimate_jacobian=True):
#     kp = torch.tensor(kp_canonical['value'])    # (bs, k, 3)
#     yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
#     t, exp = he['t'], he['tf_exp']
    
#     exp = exp.view(exp.shape[0], -1, 3)

#     kp = kp + exp
        
#     yaw = headpose_pred_to_degree(yaw)
#     pitch = headpose_pred_to_degree(pitch)
#     roll = headpose_pred_to_degree(roll)

#     rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
#     # keypoint rotation
#     kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

#     # keypoint translation
#     t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
#     kp_t = kp_rotated + t

#     kp_transformed = kp_t

#     if estimate_jacobian:
#         jacobian = kp_canonical['jacobian']   # (bs, k ,3, 3)
#         jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
#     else:
#         jacobian_transformed = None

#     return {'value': kp_transformed, 'jacobian': jacobian_transformed}

class GeneratorFullModelWithRefHe(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, he_estimator, generator, discriminator, train_params, he_estimator_ref=None, estimate_jacobian=True):
        super(GeneratorFullModelWithRefHe, self).__init__()
        self.kp_extractor = kp_extractor
        self.he_estimator = he_estimator
        self.generator = generator
        self.discriminator = discriminator
        self.he_estimator_ref = he_estimator_ref
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.estimate_jacobian = estimate_jacobian

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        if self.loss_weights['headpose'] != 0:
            self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            print('Loading hopenet')
            hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
            self.hopenet.load_state_dict(hopenet_state_dict)
            if torch.cuda.is_available():
                self.hopenet = self.hopenet.cuda()
                self.hopenet.eval()


    def forward(self, x):
        kp_canonical = self.kp_extractor(x['source'])     # {'value': value, 'jacobian': jacobian}  

        he_source = self.he_estimator(x['source'])
        he_driving = self.he_estimator(x['driving'])

        if self.he_estimator_ref is not None:
            he_source_ref = self.he_estimator_ref(x['source'])
            he_driving_ref = self.he_estimator_ref(x['driving'])
            del he_source_ref['exp']
            del he_driving_ref['exp']
            he_source.update(he_source_ref)
            he_driving.update(he_driving_ref) 

        kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
        kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian)

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])

            transformed_he_driving = self.he_estimator(transformed_frame)

            transformed_kp = keypoint_transformation(kp_canonical, transformed_he_driving, self.estimate_jacobian)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                # project 3d -> 2d
                kp_driving_2d = kp_driving['value'][:, :, :2]
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                # project 3d -> 2d
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                    transformed_jacobian_2d)
                
                jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                normed_driving = torch.inverse(jacobian_2d)
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['keypoint'] != 0:
            # print(kp_driving['value'].shape)     # (bs, k, 3)
            value_total = 0
            for i in range(kp_driving['value'].shape[1]):
                for j in range(kp_driving['value'].shape[1]):
                    dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                    dist = 0.1 - dist      # set Dt = 0.1
                    dd = torch.gt(dist, 0) 
                    value = (dist * dd).mean()
                    value_total += value

            # kp_mean_depth = kp_driving['value'][:, :, -1].mean(-1)
            # value_depth = torch.abs(kp_mean_depth - 0.33).mean()          # set Zt = 0.33

            # value_total += value_depth
            loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

        if self.loss_weights['headpose'] != 0:
            transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            driving_224 = transform_hopenet(x['driving'])

            yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

            yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)

            value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
            loss_values['headpose'] = self.loss_weights['headpose'] * value

        if self.loss_weights['expression'] != 0:
            value = torch.norm(he_driving['exp'], p=1, dim=-1).mean()
            loss_values['expression'] = self.loss_weights['expression'] * value

        return loss_values, generated

class DiscriminatorFullModelWithRefHe(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):
        super(DiscriminatorFullModelWithRefHe, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        pyramide_real = self.pyramid(torch.cat([x['driving_mesh']['_mesh_img_sec'].cuda(), x['driving']], dim=1))
        pyramide_generated = self.pyramid(torch.cat([x['driving_mesh']['mesh_img_sec'].cuda(), generated['prediction'].detach()], dim=1))
        
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        return loss_values




class GeneratorFullModelWithTF(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, hie_estimator, generator, discriminator, train_params, estimate_jacobian=True):
        super(GeneratorFullModelWithTF, self).__init__()
        self.hie_estimator = hie_estimator
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.estimate_jacobian = estimate_jacobian

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        if self.loss_weights['headpose'] != 0:
            self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            print('Loading hopenet')
            hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
            self.hopenet.load_state_dict(hopenet_state_dict)
            if torch.cuda.is_available():
                self.hopenet = self.hopenet.cuda()
                self.hopenet.eval()


    def forward(self, x):
        hie_source = self.hie_estimator(x['source'], x['source'])
        hie_driving = self.hie_estimator(x['source'], x['driving'])

        kp_canonical = {'value': hie_source['id']}    # {'value': value, 'jacobian': jacobian}   

        kp_source = keypoint_transformation(kp_canonical, hie_source, self.estimate_jacobian, exp_first=True)
        kp_driving = keypoint_transformation(kp_canonical, hie_driving, self.estimate_jacobian, exp_first=True)

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])

            transformed_hie_driving = self.hie_estimator(x['source'], transformed_frame)

            transformed_kp = keypoint_transformation(kp_canonical, transformed_hie_driving, self.estimate_jacobian, exp_first=True)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                # project 3d -> 2d
                kp_driving_2d = kp_driving['value'][:, :, :2]
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                # project 3d -> 2d
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                    transformed_jacobian_2d)
                
                jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                normed_driving = torch.inverse(jacobian_2d)
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['keypoint'] != 0:
            # print(kp_driving['value'].shape)     # (bs, k, 3)
            value_total = 0
            for i in range(kp_driving['value'].shape[1]):
                for j in range(kp_driving['value'].shape[1]):
                    dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                    dist = 0.1 - dist      # set Dt = 0.1
                    dd = torch.gt(dist, 0) 
                    value = (dist * dd).mean()
                    value_total += value

            kp_mean_depth = kp_driving['value'][:, :, -1].mean(-1)
            value_depth = torch.abs(kp_mean_depth - 0.33).mean()          # set Zt = 0.33

            value_total += value_depth
            loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

        if self.loss_weights['headpose'] != 0:
            transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            driving_224 = transform_hopenet(x['driving'])

            yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

            yaw, pitch, roll = hie_driving['yaw'], hie_driving['pitch'], hie_driving['roll']
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)

            value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
            loss_values['headpose'] = self.loss_weights['headpose'] * value

        if self.loss_weights['expression'] != 0:
            value = torch.norm(hie_driving['exp'], p=1, dim=-1).mean()
            loss_values['expression'] = self.loss_weights['expression'] * value

        return loss_values, generated


class DiscriminatorFullModelWithTF(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):
        super(DiscriminatorFullModelWithTF, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        return loss_values


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        return loss_values



class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, he_estimator, generator, discriminator, train_params, estimate_jacobian=True):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.he_estimator = he_estimator
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.estimate_jacobian = estimate_jacobian

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        if self.loss_weights['headpose'] != 0:
            self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            print('Loading hopenet')
            hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
            self.hopenet.load_state_dict(hopenet_state_dict)
            if torch.cuda.is_available():
                self.hopenet = self.hopenet.cuda()
                self.hopenet.eval()


    def forward(self, x):
        kp_canonical = self.kp_extractor(x['source'])     # {'value': value, 'jacobian': jacobian}   

        he_source = self.he_estimator(x['source'])        # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
        he_driving = self.he_estimator(x['driving'])      # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}


        # {'value': value, 'jacobian': jacobian}
        kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
        kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian)

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            
            
            ansformed_frame = transform.transform_frame(x['driving'])

            transformed_he_driving = self.he_estimator(transformed_frame)

            transformed_kp = keypoint_transformation(kp_canonical, transformed_he_driving, self.estimate_jacobian)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                # project 3d -> 2d
                kp_driving_2d = kp_driving['value'][:, :, :2]
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                # project 3d -> 2d
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                    transformed_jacobian_2d)
                
                jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                normed_driving = torch.inverse(jacobian_2d)
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['keypoint'] != 0:
            # print(kp_driving['value'].shape)     # (bs, k, 3)
            value_total = 0
            for i in range(kp_driving['value'].shape[1]):
                for j in range(kp_driving['value'].shape[1]):
                    dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                    dist = 0.1 - dist      # set Dt = 0.1
                    dd = torch.gt(dist, 0) 
                    value = (dist * dd).mean()
                    value_total += value

            kp_mean_depth = kp_driving['value'][:, :, -1].mean(-1)
            value_depth = torch.abs(kp_mean_depth - 0.33).mean()          # set Zt = 0.33

            value_total += value_depth
            loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

        if self.loss_weights['headpose'] != 0:
            transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            driving_224 = transform_hopenet(x['driving'])

            yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

            yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)

            value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
            loss_values['headpose'] = self.loss_weights['headpose'] * value

        if self.loss_weights['expression'] != 0:
            value = torch.norm(he_driving['exp'], p=1, dim=-1).mean()
            loss_values['expression'] = self.loss_weights['expression'] * value

        return loss_values, generated

class GeneratorFullModelWithSeg(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, he_estimator, generator, discriminator, train_params, he_estimator_ref=None, estimate_jacobian=True):
        super(GeneratorFullModelWithSeg, self).__init__()
        self.kp_extractor = kp_extractor
        self.he_estimator = he_estimator
        self.generator = generator
        self.discriminator = discriminator
        self.he_estimator_ref = he_estimator_ref
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        self.pyramid_cond = ImagePyramide(self.scales, generator.image_channel + 1)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
            self.pyramid_cond = self.pyramid_cond.cuda()
            
        self.loss_weights = train_params['loss_weights']

        self.estimate_jacobian = estimate_jacobian

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        if self.loss_weights['headpose'] != 0:
            self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            print('Loading hopenet')
            hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
            self.hopenet.load_state_dict(hopenet_state_dict)
            if torch.cuda.is_available():
                self.hopenet = self.hopenet.cuda()
                self.hopenet.eval()


    def forward(self, x):
        kp_canonical = self.kp_extractor(x['source'])     # {'value': value, 'jacobian': jacobian}  

        he_source = self.he_estimator(x['source'])
        he_driving = self.he_estimator(x['driving'])

        if self.he_estimator_ref is not None:
            he_source_ref = self.he_estimator_ref(x['source'])
            he_driving_ref = self.he_estimator_ref(x['driving'])
            del he_source_ref['exp']
            del he_driving_ref['exp']
            he_source.update(he_source_ref)
            he_driving.update(he_driving_ref) 

        kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
        kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian)

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        if self.loss_weights['motion_match'] != 0:
            motion = generated['deformation'] # B x d x h x w x 3
            motion = motion.permute(0, 4, 1, 2, 3) # B x 3 x d x h x w
            it_section = x['driving_mesh']['raw_value'] # B x N x 3
            motion_GT = x['source_mesh']['raw_value'] # B x N x 3
            it_section_eye = it_section[:,  x['source_mesh']['MP_EYE_SECTIONS'][0].long()]
            motion_GT_eye = motion_GT[:,  x['source_mesh']['MP_EYE_SECTIONS'][0].long()]
            it_section_mouth = it_section[:,  x['source_mesh']['MP_MOUTH_SECTIONS'][0].long()]
            motion_GT_mouth = motion_GT[:,  x['source_mesh']['MP_MOUTH_SECTIONS'][0].long()]
            # print(f'it section shape: {it_section.shape}')
            # print(f'motion_GT section shape: {motion_GT.shape}')
            # print(f'motion shape: {motion.shape}')
            motion_section = F.grid_sample(motion, it_section[:, :, None, None])
            motion_section = motion_section.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

            motion_section_eye = F.grid_sample(motion, it_section_eye[:, :, None, None])
            motion_section_eye = motion_section_eye.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

            motion_section_mouth = F.grid_sample(motion, it_section_mouth[:, :, None, None])
            motion_section_mouth = motion_section_mouth.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

            # print(f'motion Gt size {motion_GT.shape}')
            # print(f'motion size {motion_section.shape}')
            # print(f'motion_section: {motion_section}')
            # print(f'motion_section_GT : {motion_GT}')
            
            loss_values['motion_match'] = 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section, motion_GT) \
                                            + 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section_eye, motion_GT_eye) \
                                            + 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section_mouth, motion_GT_mouth)

        if self.loss_weights['generator_gan'] != 0:
            pyramide_real = self.pyramid_cond(torch.cat([x['driving_mesh']['_mesh_img_sec'].cuda(), x['driving']], dim=1))
            pyramide_generated = self.pyramid_cond(torch.cat([x['driving_mesh']['mesh_img_sec'].cuda(), generated['prediction']], dim=1))
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])

            transformed_he_driving = self.he_estimator(transformed_frame)

            transformed_kp = keypoint_transformation(kp_canonical, transformed_he_driving, self.estimate_jacobian)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                # project 3d -> 2d
                kp_driving_2d = kp_driving['value'][:, :, :2]
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                # project 3d -> 2d
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                    transformed_jacobian_2d)
                
                jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                normed_driving = torch.inverse(jacobian_2d)
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['keypoint'] != 0:
            # print(kp_driving['value'].shape)     # (bs, k, 3)
            value_total = 0
            for i in range(kp_driving['value'].shape[1]):
                for j in range(kp_driving['value'].shape[1]):
                    dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                    dist = 0.1 - dist      # set Dt = 0.1
                    dd = torch.gt(dist, 0) 
                    value = (dist * dd).mean()
                    value_total += value

            kp_mean_depth = kp_driving['value'][:, :, -1].mean(-1)
            value_depth = torch.abs(kp_mean_depth - 0.33).mean()          # set Zt = 0.33

            value_total += value_depth
            loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

        if self.loss_weights['headpose'] != 0:
            transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            driving_224 = transform_hopenet(x['driving'])

            yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

            yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)

            value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
            loss_values['headpose'] = self.loss_weights['headpose'] * value

        if self.loss_weights['expression'] != 0:
            value = torch.norm(he_driving['exp'], p=1, dim=-1).mean()
            loss_values['expression'] = self.loss_weights['expression'] * value

        return loss_values, generated




class DiscriminatorFullModelWithSeg(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):
        super(DiscriminatorFullModelWithSeg, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel + 1)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        # pyramide_real = self.pyramid(x['driving'])
        # pyramide_generated = self.pyramid(generated['prediction'].detach())

        pyramide_real = self.pyramid(torch.cat([x['driving_mesh']['_mesh_img_sec'].cuda(), x['driving']], dim=1))
        pyramide_generated = self.pyramid(torch.cat([x['driving_mesh']['mesh_img_sec'].cuda(), generated['prediction'].detach()], dim=1))
        
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        return loss_values


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        return loss_values


class ExpTransformerTrainer(GeneratorFullModelWithSeg):
    def __init__(self, stage, exp_transformer, kp_extractor, he_estimator, generator, discriminator, train_params, estimate_jacobian=True, device_ids=[0]):
        super(ExpTransformerTrainer, self).__init__(kp_extractor, he_estimator, generator, discriminator, train_params, estimate_jacobian=estimate_jacobian)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        # self.train()
        
        self.exp_transformer = exp_transformer
        
        exp_transformer.train()
        
        discriminator.train()
        for p in discriminator.parameters():
            p.requires_grad = True

        generator.train()
        for p in generator.parameters():
            p.requires_grad = True

        self.stage = stage


        # self.sections = train_params['sections']
        # self.split_ids = [sec[1] for sec in self.sections]

        if self.stage == 2:
            for name, p in self.exp_transformer.named_parameters():
                if 'delta' not in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            self.exp_transformer.train()

            generator.eval()
            for p in generator.parameters():
                p.requires_grad = False
            
        #     self.id_classifier_scale = train_params['id_classifier_scale']
        #     self.id_classifier_scaler = AntiAliasInterpolation2d(3, self.id_classifier_scale).to(device_ids[0])
        #     self.id_classifier = InceptionResnetV1(pretrained='vggface2').eval().to(device_ids[0])

        self.log_loss = lambda x: -torch.log((1 + x).clamp(min=1e-6)).mean()

    def forward(self, x, cycled_drive=False):
        if self.stage == 1:
            loss_values = {}
            
            bs = len(x['source'])

            source_mesh = x['source_mesh']
            driving_mesh = x['driving_mesh']
            
            tf_output = self.exp_transformer({'mesh': source_mesh['value']}, {'mesh': driving_mesh['value']}, placeholder=['kp'])

            kp_canonical = {'value': tf_output['src_kp']}
            kp_canonical_drv = {'value': tf_output['drv_kp']}

            
            # {'value': value, 'jacobian': jacobian}
            kp_source = keypoint_transformation(kp_canonical, source_mesh)
            kp_driving = keypoint_transformation(kp_canonical_drv, driving_mesh)

            kp_source['_mesh_img_sec'] = x['source_mesh']['_mesh_img_sec']
            kp_source['mesh_img_sec'] = x['source_mesh']['mesh_img_sec']
            kp_driving['_mesh_img_sec'] = x['driving_mesh']['_mesh_img_sec']
            kp_driving['mesh_img_sec'] = x['driving_mesh']['mesh_img_sec']
            

            
            generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
            generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
            pyramide_real = self.pyramid(x['driving'])
            pyramide_generated = self.pyramid(generated['prediction'])
            
            generated['source_kp_canonical'] = {'value': kp_source['canonical']}
            generated['driving_kp_canonical'] = {'value': kp_driving['canonical']}
            generated['source_mesh_image'] = kp_source['mesh_img_sec']
            generated['driving_mesh_image'] = kp_driving['mesh_img_sec']
            
            # if cycled_drive:
            #     ## cycled expression drive
            #     src_style = tf_output['src_embedding']['style']
            #     src_exp_code_decoded = tf_output['drv_embedding']['exp']
            #     src_exp_code_cycled_decoded = torch.cat([src_exp_code_decoded[1:], src_exp_code_decoded[[0]]], dim=0)
            #     cycled_embedding = {'style': src_style, 'exp': src_exp_code_cycled_decoded}
            #     src_exp_cycled = self.exp_transformer.decode(cycled_embedding)['exp']

            #     source_mesh_cycled = {'U': torch.cat([driving_mesh['U'][1:], driving_mesh['U'][[0]]], dim=0), 'scale': torch.cat([driving_mesh['scale'][1:], driving_mesh['scale'][[0]]], dim=0), 'exp': src_exp_cycled}
 
            #     kp_source_cycled = keypoint_transformation(kp_canonical, source_mesh_cycled)

            #     generated_cycled = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_source_cycled)
            #     for k, v in list(generated_cycled.items()):
            #         generated[f'{k}_cycled'] = v
            #     generated['kp_driving_cycled'] = kp_source_cycled
                

            if self.loss_weights['log'] != 0:
                src_exp_code = tf_output['src_embedding']['delta_exp_code']   # B x num_heads
                drv_exp_code = tf_output['drv_embedding']['delta_exp_code']   # B x num_heads
                greater_mask = (src_exp_code >= drv_exp_code).detach() 
                less_mask = ~greater_mask
                greater_labels = torch.cat([src_exp_code[greater_mask], drv_exp_code[less_mask]], dim=0)
                less_labels = torch.cat([src_exp_code[less_mask], drv_exp_code[greater_mask]], dim=0)
                loss_values['log'] = self.loss_weights['log'] * (self.log_loss(greater_labels) + self.log_loss(-less_labels))

            if self.loss_weights['l1'] != 0:
                loss_values['l1'] = self.loss_weights['l1'] * F.l1_loss(generated['prediction'], x['driving'])

            if self.loss_weights['motion_match'] != 0:
                motion = generated['deformation'] # B x d x h x w x 3
                motion = motion.permute(0, 4, 1, 2, 3) # B x 3 x d x h x w
                it_section = driving_mesh['raw_value'] # B x N x 3
                motion_GT = source_mesh['raw_value'] # B x N x 3
                it_section_eye = it_section[:, source_mesh['OPENFACE_EYE_IDX'][0].long()]
                motion_GT_eye = motion_GT[:, source_mesh['OPENFACE_EYE_IDX'][0].long()]
                it_section_mouth = it_section[:, source_mesh['OPENFACE_LIP_IDX'][0].long()]
                motion_GT_mouth = motion_GT[:, source_mesh['OPENFACE_LIP_IDX'][0].long()]
                # print(f'it section shape: {it_section.shape}')
                # print(f'motion_GT section shape: {motion_GT.shape}')
                # print(f'motion shape: {motion.shape}')
                motion_section = F.grid_sample(motion, it_section[:, :, None, None])
                motion_section = motion_section.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

                motion_section_eye = F.grid_sample(motion, it_section_eye[:, :, None, None])
                motion_section_eye = motion_section_eye.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

                motion_section_mouth = F.grid_sample(motion, it_section_mouth[:, :, None, None])
                motion_section_mouth = motion_section_mouth.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

                # print(f'motion Gt size {motion_GT.shape}')
                # print(f'motion size {motion_section.shape}')
                # print(f'motion_section: {motion_section}')
                # print(f'motion_section_GT : {motion_GT}')
                
                loss_values['motion_match'] = 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section, motion_GT) \
                                                + 10 * self.loss_weights['motion_match'] * F.l1_loss(motion_section_eye, motion_GT_eye) \
                                                + 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section_mouth, motion_GT_mouth)

            if np.array(self.loss_weights['localized']).sum() != 0:
                localized_loss = 0
                split = []
                centre = []
                ws = []

                for sec, w in zip(self.sections, self.loss_weights['localized']):
                    center = kp_source['value'][:, sec[0]].mean(dim=1)
                    centre.append(center)
                    split.append(sec[1])
                    ws.append(w)

                split = kp_source['prior'].split(split, dim=1) # [num_sec] x B x len_sec x 3

                for sec, center, w in zip(split, centre, ws):
                    localized_loss = localized_loss + w * (sec - center.unsqueeze(1)).norm() / (sec.size(0) * sec.size(1))
                
                loss_values['localized'] = localized_loss

            if sum(self.loss_weights['perceptual']) != 0:
                value_total = 0
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

            if self.loss_weights['generator_gan'] != 0:
                pyramide_real = self.pyramid_cond(torch.cat([kp_driving['_mesh_img_sec'].cuda(), x['driving']], dim=1))
                pyramide_generated = self.pyramid_cond(torch.cat([kp_driving['mesh_img_sec'].cuda(), generated['prediction']], dim=1))
                discriminator_maps_generated = self.discriminator(pyramide_generated)
                discriminator_maps_real = self.discriminator(pyramide_real)
                
                value_total = 0
                for scale in self.disc_scales:
                    key = 'prediction_map_%s' % scale
                    if self.train_params['gan_mode'] == 'hinge':
                        value = -torch.mean(discriminator_maps_generated[key])
                    elif self.train_params['gan_mode'] == 'ls':
                        value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    else:
                        raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                    value_total += self.loss_weights['generator_gan'] * value
                loss_values['gen_gan'] = value_total

                if sum(self.loss_weights['feature_matching']) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.loss_weights['feature_matching'][i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.loss_weights['feature_matching'][i] * value
                        loss_values['feature_matching'] = value_total

            if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
                transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
                transformed_frame = transform.transform_frame(x['driving'])

                transformed_embedding = self.exp_transformer.encode(transformed_frame) 
                exp_transformed = self.exp_transformer.decode({'style': tf_output['src_embedding']['style'], 'exp': transformed_embedding['exp']})

                transformed_kp = {'value': kp_driving['prior'] - kp_driving['exp'] + exp_transformed['exp']}

                generated['transformed_frame'] = transformed_frame
                generated['transformed_kp'] = transformed_kp

                ## Value loss part
                if self.loss_weights['equivariance_value'] != 0:
                    # project 3d -> 2d
                    kp_driving_2d = kp_driving['prior'][:, :, :2]
                    transformed_kp_2d = transformed_kp['value'][:, :, :2]
                    value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                    loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

                ## jacobian loss part
                if self.loss_weights['equivariance_jacobian'] != 0:
                    # project 3d -> 2d
                    transformed_kp_2d = transformed_kp['value'][:, :, :2]
                    transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                    jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                        transformed_jacobian_2d)
                    
                    jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                    normed_driving = torch.inverse(jacobian_2d)
                    normed_transformed = jacobian_transformed
                    value = torch.matmul(normed_driving, normed_transformed)

                    eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                    value = torch.abs(eye - value).mean()
                    loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

            if self.loss_weights['keypoint'] != 0:
                # print(kp_driving['value'].shape)     # (bs, k, 3)
                value_total = 0
                for i in range(kp_canonical['value'].shape[1]):
                    for j in range(kp_canonical['value'].shape[1]):
                        dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                        dist = 0.2 - dist      # set Dt = 0.1
                        dd = torch.gt(dist, 0) 
                        value = (dist * dd).mean()
                        value_total += value

                kp_mean_depth = kp_canonical['value'][:, :, -1].mean(-1)
                value_depth = torch.abs(kp_mean_depth + 0.33).mean()          # set Zt = 0.33
                # print(f'kp_mean_depth: {kp_driving["value"][:, :, -1].mean(-1)}')
                # print(f'kp_depth: {kp_driving["value"][:, :, -1]}')
                value_total += value_depth
                loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

            if self.loss_weights['headpose'] != 0:
                transform_hopenet =  transforms.Compose([
                                                        transforms.Resize(size=(224, 224)),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        transforms.ToTensor()])
                
                # print(f'driving image shape: {x["driving"][0].cpu().size()}')
                # print(f'driving image shape: {transforms.ToPILImage()(x["driving"][0].permute(1, 2, 0).cpu()).size}')
                # print(f'driving image: {transforms.ToPILImage()(x["driving"][0].permute(1, 2, 0).cpu())}')
                driving_224 = transform_hopenet(x['driving'].cpu()).cuda()
                driving_224 = x['hopenet_driving']

                yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
                yaw_gt = headpose_pred_to_degree(yaw_gt)
                pitch_gt = headpose_pred_to_degree(pitch_gt)
                roll_gt = headpose_pred_to_degree(roll_gt)

                yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
                yaw = headpose_pred_to_degree(yaw)
                pitch = headpose_pred_to_degree(pitch)
                roll = headpose_pred_to_degree(roll)

                value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
                loss_values['headpose'] = self.loss_weights['headpose'] * value

            if self.loss_weights['expression'] != 0:
                value = torch.norm(driving_mesh['exp'], p=1, dim=-1).mean()
                loss_values['expression'] = self.loss_weights['expression'] * value

        elif self.stage == 2:
            loss_values = {}
            
            bs = len(x['source'])
            
            # kp_canonical = self.kp_extractor(x['source'])     # {'value': value, 'jacobian': jacobian}   

            src_feat = self.he_estimator(x['source'])        # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
            drv_feat = self.he_estimator(x['driving'])      # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
            
            source_mesh = x['source_mesh']
            driving_mesh = x['driving_mesh']
            
            # driving_mesh['scale'] = source_mesh['scale']
            # driving_mesh['U'] = np.source_mesh['U']
            
            tf_output = self.exp_transformer({'feat': src_feat, 'mesh': source_mesh['value']}, {'feat': drv_feat, 'mesh': driving_mesh['value']})

            src_exp = tf_output['src_exp']
            drv_exp = tf_output['drv_exp']


            kp_canonical = {'value': tf_output['src_kp']}
            kp_canonical_drv = {'value': tf_output['drv_kp']}

            delta_src_embedding = {'delta_style_code': tf_output['src_embedding']['delta_style_code'], 'delta_exp_code': tf_output['src_embedding']['delta_exp_code']}
            delta_drv_embedding = {'delta_style_code': tf_output['src_embedding']['delta_style_code'], 'delta_exp_code': tf_output['drv_embedding']['delta_exp_code']}
            # {'value': value, 'jacobian': jacobian}
            
            src_delta = self.exp_transformer.decode(delta_src_embedding)['delta']
            drv_delta = self.exp_transformer.decode(delta_drv_embedding)['delta']
            
            driving_mesh['delta'] = -src_delta + drv_delta
            
            kp_source = keypoint_transformation(kp_canonical, source_mesh)
            kp_driving = keypoint_transformation(kp_canonical, driving_mesh)

            kp_source['_mesh_img_sec'] = x['source_mesh']['_mesh_img_sec']
            kp_source['mesh_img_sec'] = x['source_mesh']['mesh_img_sec']
            kp_driving['_mesh_img_sec'] = x['driving_mesh']['_mesh_img_sec']
            kp_driving['mesh_img_sec'] = x['driving_mesh']['mesh_img_sec']
            # self.denormalize(kp_source, x['source'])        # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 's_e': s_e}
            # self.denormalize(kp_driving, x['driving'])      # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 's_e': s_e}

            
            # driving_224 = x['hopenet_driving']
            # yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)

            # reg = self.regularize(kp_canonical_source, kp_canonical_driving) # regularizor loss


            
            # x_reg = kp_canonical['value'].flatten(1)
            # for x_i in x_reg:
            #     self.pca_x.register(x_i.detach().cpu())
            # if self.pca_x.steps > 2:
            #     mu_x, u_x, s_x = self.pca_x.get_state(device='cuda')
            #     loss_reg = ((x_reg - mu_x[None]).unsqueeze(1) @ ((u_x @ (s_x ** 2) @ u_x.t())[None] + self.sigma_err[None]).inverse() @ (x_reg - mu_x[None]).unsqueeze(2)).mean() # 1
            #     loss['regularizor'] = self.loss_weights['regularizor'] * loss_reg
            #     self.mu_x, self.u_x, self.s_x = self.pca_x.get_state()
            # else:
            #     loss['regularizor'] = self.loss_weights['regularizor'] * torch.zeros(1).cuda().mean()

            # if self.pca_x.steps * self.pca_e.steps > 0:
            #     kp_source, kp_driving = reg['kp_source'], reg['kp_driving']
            #     loss = {k: self.loss_weights[k] * v for k, v in reg['loss'].items()}
            # else:
            #     kp_source, kp_driving = kp_canonical_source, kp_canonical_driving
            #     loss = {k: self.loss_weights[k] * torch.zeros(1).cuda() for k, v in reg['loss'].items()}


            # {'value': value, 'jacobian': jacobian}
            # kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
            # kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian)

            
            # print('entering generator')
            # print(f'kp_source value: {kp_source["value"]}')
            # print(f'kp_driving value: {kp_driving["value"]}')
            
            generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
            generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
            pyramide_real = self.pyramid(x['driving'])
            pyramide_generated = self.pyramid(generated['prediction'])
            
            generated['source_kp_canonical'] = {'value': kp_source['canonical']}
            generated['driving_kp_canonical'] = {'value': kp_driving['canonical']}
            generated['source_mesh_image'] = kp_source['mesh_img_sec']
            generated['driving_mesh_image'] = kp_driving['mesh_img_sec']
            
            if cycled_drive:
                ## cycled expression drive
                delta_style_code = tf_output['src_embedding']['delta_style_code']
                delta_exp_code = tf_output['src_embedding']['delta_exp_code']
                delta_exp_code_cycled = torch.cat([delta_exp_code[1:], delta_exp_code[[0]]], dim=0)
                src_delta_cycled = self.exp_transformer.decode({'delta_style_code': delta_style_code, 'delta_exp_code': delta_exp_code_cycled})['delta']
                source_mesh_cycled = {'U': source_mesh['U'], 'scale': source_mesh['scale'], 'delta': - src_delta + src_delta_cycled}

                kp_source_cycled = keypoint_transformation(kp_canonical, source_mesh_cycled)

                generated_cycled = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_source_cycled)
                for k, v in list(generated_cycled.items()):
                    generated[f'{k}_cycled'] = v
                generated['kp_driving_cycled'] = kp_source_cycled
                

            if self.loss_weights['log'] != 0:
                src_exp_code = tf_output['src_embedding']['delta_exp_code']   # B x num_heads
                drv_exp_code = tf_output['drv_embedding']['delta_exp_code']   # B x num_heads
                greater_mask = (src_exp_code >= drv_exp_code).detach() 
                less_mask = ~greater_mask
                greater_labels = torch.cat([src_exp_code[greater_mask], drv_exp_code[less_mask]], dim=0)
                less_labels = torch.cat([src_exp_code[less_mask], drv_exp_code[greater_mask]], dim=0)
                loss_values['log'] = self.loss_weights['log'] * (self.log_loss(greater_labels) + self.log_loss(-less_labels))

            if self.loss_weights['style'] != 0:
                src_style_code = tf_output['src_embedding']['delta_style_code']   # B x num_heads
                drv_style_code = tf_output['drv_embedding']['delta_style_code']   # B x num_heads
                loss_values['style'] = self.loss_weights['style'] * torch.norm(src_style_code - drv_style_code, dim=1) ** 2
            
            
            if self.loss_weights['l1'] != 0:
                loss_values['l1'] = self.loss_weights['l1'] * F.l1_loss(generated['prediction'], x['driving'])

            if self.loss_weights['motion_match'] != 0:
                motion = generated['deformation'] # B x d x h x w x 3
                motion = motion.permute(0, 4, 1, 2, 3) # B x 3 x d x h x w
                it_section = driving_mesh['raw_value'] # B x N x 3
                motion_GT = source_mesh['raw_value'] # B x N x 3
                it_section_eye = it_section[:, source_mesh['OPENFACE_EYE_IDX'][0].long()]
                motion_GT_eye = motion_GT[:, source_mesh['OPENFACE_EYE_IDX'][0].long()]
                it_section_mouth = it_section[:, source_mesh['OPENFACE_LIP_IDX'][0].long()]
                motion_GT_mouth = motion_GT[:, source_mesh['OPENFACE_LIP_IDX'][0].long()]
                # print(f'it section shape: {it_section.shape}')
                # print(f'motion_GT section shape: {motion_GT.shape}')
                # print(f'motion shape: {motion.shape}')
                motion_section = F.grid_sample(motion, it_section[:, :, None, None])
                motion_section = motion_section.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

                motion_section_eye = F.grid_sample(motion, it_section_eye[:, :, None, None])
                motion_section_eye = motion_section_eye.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

                motion_section_mouth = F.grid_sample(motion, it_section_mouth[:, :, None, None])
                motion_section_mouth = motion_section_mouth.squeeze(4).squeeze(3).transpose(1,2) # B x N x 3

                # print(f'motion Gt size {motion_GT.shape}')
                # print(f'motion size {motion_section.shape}')
                # print(f'motion_section: {motion_section}')
                # print(f'motion_section_GT : {motion_GT}')
                
                loss_values['motion_match'] = 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section, motion_GT) \
                                                + 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section_eye, motion_GT_eye) \
                                                + 1 * self.loss_weights['motion_match'] * F.l1_loss(motion_section_mouth, motion_GT_mouth)

            if np.array(self.loss_weights['localized']).sum() != 0:
                localized_loss = 0
                split = []
                centre = []
                ws = []

                for sec, w in zip(self.sections, self.loss_weights['localized']):
                    center = kp_source['value'][:, sec[0]].mean(dim=1)
                    centre.append(center)
                    split.append(sec[1])
                    ws.append(w)

                split = kp_source['prior'].split(split, dim=1) # [num_sec] x B x len_sec x 3

                for sec, center, w in zip(split, centre, ws):
                    localized_loss = localized_loss + w * (sec - center.unsqueeze(1)).norm() / (sec.size(0) * sec.size(1))
                
                loss_values['localized'] = localized_loss

            if sum(self.loss_weights['perceptual']) != 0:
                value_total = 0
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

            if self.loss_weights['generator_gan'] != 0:
                pyramide_real = self.pyramid_cond(torch.cat([kp_driving['_mesh_img_sec'].cuda(), x['driving']], dim=1))
                pyramide_generated = self.pyramid_cond(torch.cat([kp_driving['mesh_img_sec'].cuda(), generated['prediction']], dim=1))
                discriminator_maps_generated = self.discriminator(pyramide_generated)
                discriminator_maps_real = self.discriminator(pyramide_real)
                
                value_total = 0
                for scale in self.disc_scales:
                    key = 'prediction_map_%s' % scale
                    if self.train_params['gan_mode'] == 'hinge':
                        value = -torch.mean(discriminator_maps_generated[key])
                    elif self.train_params['gan_mode'] == 'ls':
                        value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    else:
                        raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                    value_total += self.loss_weights['generator_gan'] * value
                loss_values['gen_gan'] = value_total

                if sum(self.loss_weights['feature_matching']) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.loss_weights['feature_matching'][i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.loss_weights['feature_matching'][i] * value
                        loss_values['feature_matching'] = value_total

            if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
                transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
                transformed_frame = transform.transform_frame(x['driving'])

                transformed_embedding = self.exp_transformer.encode(transformed_frame) 
                exp_transformed = self.exp_transformer.decode({'style': tf_output['src_embedding']['style'], 'exp': transformed_embedding['exp']})

                transformed_kp = {'value': kp_driving['prior'] - kp_driving['exp'] + exp_transformed['exp']}

                generated['transformed_frame'] = transformed_frame
                generated['transformed_kp'] = transformed_kp

                ## Value loss part
                if self.loss_weights['equivariance_value'] != 0:
                    # project 3d -> 2d
                    kp_driving_2d = kp_driving['prior'][:, :, :2]
                    transformed_kp_2d = transformed_kp['value'][:, :, :2]
                    value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                    loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

                ## jacobian loss part
                if self.loss_weights['equivariance_jacobian'] != 0:
                    # project 3d -> 2d
                    transformed_kp_2d = transformed_kp['value'][:, :, :2]
                    transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                    jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                        transformed_jacobian_2d)
                    
                    jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                    normed_driving = torch.inverse(jacobian_2d)
                    normed_transformed = jacobian_transformed
                    value = torch.matmul(normed_driving, normed_transformed)

                    eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                    value = torch.abs(eye - value).mean()
                    loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

            if self.loss_weights['keypoint'] != 0:
                # print(kp_driving['value'].shape)     # (bs, k, 3)
                value_total = 0
                for i in range(kp_canonical['value'].shape[1]):
                    for j in range(kp_canonical['value'].shape[1]):
                        dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                        dist = 0.2 - dist      # set Dt = 0.1
                        dd = torch.gt(dist, 0) 
                        value = (dist * dd).mean()
                        value_total += value

                kp_mean_depth = kp_canonical['value'][:, :, -1].mean(-1)
                value_depth = torch.abs(kp_mean_depth + 0.33).mean()          # set Zt = 0.33
                # print(f'kp_mean_depth: {kp_driving["value"][:, :, -1].mean(-1)}')
                # print(f'kp_depth: {kp_driving["value"][:, :, -1]}')
                value_total += value_depth
                loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

            if self.loss_weights['headpose'] != 0:
                transform_hopenet =  transforms.Compose([
                                                        transforms.Resize(size=(224, 224)),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        transforms.ToTensor()])
                
                # print(f'driving image shape: {x["driving"][0].cpu().size()}')
                # print(f'driving image shape: {transforms.ToPILImage()(x["driving"][0].permute(1, 2, 0).cpu()).size}')
                # print(f'driving image: {transforms.ToPILImage()(x["driving"][0].permute(1, 2, 0).cpu())}')
                driving_224 = transform_hopenet(x['driving'].cpu()).cuda()
                driving_224 = x['hopenet_driving']

                yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
                yaw_gt = headpose_pred_to_degree(yaw_gt)
                pitch_gt = headpose_pred_to_degree(pitch_gt)
                roll_gt = headpose_pred_to_degree(roll_gt)

                yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
                yaw = headpose_pred_to_degree(yaw)
                pitch = headpose_pred_to_degree(pitch)
                roll = headpose_pred_to_degree(roll)

                value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
                loss_values['headpose'] = self.loss_weights['headpose'] * value

            if self.loss_weights['expression'] != 0:
                value = torch.norm(driving_mesh['exp'], p=1, dim=-1).mean()
                loss_values['expression'] = self.loss_weights['expression'] * value


        return loss_values, generated
