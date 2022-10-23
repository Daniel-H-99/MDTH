from torch import nn

import torch.nn.functional as F
import torch
import pickle as pkl
from natsort import natsorted
import random

import sys

import numpy as np
import math
import os
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from tqdm import tqdm
import cv2
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.transform import Rotation as R


def normalized_to_pixel_coordinates(landmark_dict, image_width, image_height):
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    landmark_pixel_coord_dict = {}

    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue

        if not (is_valid_normalized_value(coord[0]) and
                is_valid_normalized_value(coord[1])):

            return None
        x_px = coord[0] * image_width
        y_px = coord[1] * image_height
        z_px = coord[2] * image_width
        landmark_pixel_coord_dict[idx] = [x_px, y_px, z_px]
    return landmark_pixel_coord_dict

# def extract_openface_mesh(image, noise=None):
#     mesh = {}
#     H, W = image.shape[:2]
#     bb, lm = glfa(image)
#     # lm[:, 1] = H - lm[:, 1]
#     lm_normed_3d, U, Ind = nm(lm, H, W)
#     # U = torch.from_numpy(U['U'])
#     normalizer = U['normalizer']
#     U = U['U']
#     # print(f'normed: {lm_normed_3d}')
#     if noise is not None:
#         # if np.r
#         noise = np.random.randn(3, 3) * noise
#         noise = noise.clip(-0.1, 0.1) * 100
#         lm_normed_3d[[3] + list(range(17, 22)) + list(range(36, 42))] += noise[[0]]
#         lm_normed_3d[[4] + list(range(22, 27)) + list(range(42, 48))] += noise[[1]]
#         lm_normed_3d[48:] += noise[[2]]
#         # lm_normed_3d = np.concatenate([lm_normed_3d, noise], axis=0)

#     lm_3d = np.concatenate([lm_normed_3d, np.ones((len(lm_normed_3d), 1))], axis=1) @ U
#     if noise is not None:
#         noise_real = noise @ U[:3, :3]
#     else:
#         noise_real = None

#     lm_3d = lm_3d[:, :3]
#     # lm_3d[:, 1] = H - lm_3d[:, 1] 
#     scale = H // 2
#     lm_scaled_3d = lm_normed_3d / scale  
#     mesh["raw_value"] = lm_3d
#     mesh["value"] = lm_scaled_3d.astype(np.float32)
#     mesh["U"] = U.astype(np.float32)
#     mesh["scale"] = scale
#     # print(f'landmark: {lm}')
#     # print(f'mesh: {mesh}')

#     return mesh, noise_real, normalizer
    
def extract_mesh(image):
    # image: RGB, ubyte
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
            annotated_image = image.copy()
            image_rows, image_cols, _ = image.shape
            results = face_mesh.process(image)
            target_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
            target_dict = normalized_to_pixel_coordinates(target_dict, image_cols, image_rows)
            raw_mesh = landmarkdict_to_mesh_tensor(target_dict)
            # raw_mesh[:, 2] = raw_mesh[:, 2]
            raw_mesh[:, 2] = raw_mesh[:, 2] - 50
            # R, t, c = Umeyama_algorithm(reference_dict, target_dict)
            # target_dict['R'] = R
            # target_dict['t'] = t
            # target_dict['c'] = c
            # normalized_mesh = landmarkdict_to_normalized_mesh_tensor(target_dict)
            # return {'value': normalized_mesh, 'R': R, 't': t, 'c': c, 'raw_value': raw_mesh}
            return {'raw_value': raw_mesh}

KEY_VARIANCE = {17: 18.965496063232422, 84: 18.238582611083984, 16: 17.62625503540039, 314: 17.342655181884766, 85: 17.177982330322266, 315: 16.50968360900879, 15: 16.047866821289062, 181: 15.401814460754395, 86: 15.401474952697754, 14: 15.25156021118164, 316: 14.916162490844727, 87: 14.875166893005371, 180: 14.537986755371094, 317: 14.533039093017578, 405: 13.683375358581543, 179: 13.18199348449707, 404: 13.172924995422363, 178: 13.024672508239746, 18: 12.389984130859375, 402: 12.287979125976562, 403: 12.233527183532715, 83: 11.809043884277344, 313: 11.569454193115234, 91: 11.24007511138916, 90: 10.997507095336914, 200: 10.511945724487305, 88: 10.414156913757324, 89: 10.384500503540039, 201: 9.809059143066406, 199: 9.693489074707031, 421: 9.59658145904541, 320: 9.55312728881836, 318: 9.529691696166992, 321: 9.519864082336426, 152: 9.518776893615723, 175: 9.441099166870117, 319: 9.315605163574219, 182: 9.280145645141602, 406: 8.966537475585938, 377: 8.871063232421875, 148: 8.866243362426758, 208: 8.824270248413086, 428: 8.686577796936035, 171: 8.640405654907227, 396: 8.587196350097656, 95: 7.947456359863281, 96: 7.790765762329102, 77: 7.757366180419922, 400: 7.5607757568359375, 146: 7.51237154006958, 176: 7.385627746582031, 194: 7.3249640464782715, 418: 7.113580703735352, 324: 7.106655120849609, 369: 7.013779640197754, 140: 6.886379718780518, 325: 6.788877010345459, 32: 6.746204376220703, 262: 6.677685737609863, 307: 6.592177867889404, 106: 6.362393379211426, 375: 6.215060234069824, 335: 6.144772529602051, 378: 6.065235137939453, 149: 5.690601348876953, 395: 5.456660270690918, 204: 5.203995704650879, 170: 5.194314956665039, 62: 5.111909866333008, 78: 5.110922813415527, 76: 5.086575508117676, 424: 5.080007553100586, 61: 5.026874542236328, 431: 4.981131076812744, 211: 4.964018821716309, 379: 4.698431968688965, 308: 4.418610572814941, 292: 4.409973621368408, 306: 4.324930191040039, 150: 4.229571342468262, 43: 4.225931167602539, 291: 4.222723960876465, 394: 4.145513534545898, 273: 4.071269989013672, 169: 3.8565611839294434, 183: 3.514094352722168, 184: 3.5088438987731934, 430: 3.4743614196777344, 365: 3.446742534637451, 185: 3.4464504718780518, 202: 3.4442028999328613, 210: 3.3955860137939453, 191: 3.3844614028930664, 422: 3.373734474182129, 136: 2.963453769683838, 364: 2.9446029663085938, 407: 2.8978986740112305, 408: 2.84171199798584, 415: 2.8094229698181152, 409: 2.7516589164733887, 287: 2.6652674674987793, 57: 2.6370229721069336, 397: 2.6083054542541504, 135: 2.5998363494873047, 40: 2.3424015045166016, 74: 2.305135726928711, 42: 2.2818500995635986, 288: 2.213219165802002, 80: 2.2062857151031494, 172: 2.1851091384887695, 367: 2.1739962100982666, 212: 2.151231050491333, 432: 2.1447606086730957, 434: 2.1008245944976807, 214: 1.9824650287628174, 270: 1.9380097389221191, 361: 1.9151620864868164, 310: 1.9064078330993652, 272: 1.890311598777771, 304: 1.8824903964996338, 410: 1.8465478420257568, 138: 1.8052912950515747, 58: 1.7760241031646729, 186: 1.7218245267868042, 81: 1.6777923107147217, 435: 1.6668847799301147, 41: 1.661131501197815, 13: 1.654787302017212, 73: 1.641161561012268, 39: 1.6315116882324219, 323: 1.629990577697754, 82: 1.6037687063217163, 312: 1.5759391784667969, 416: 1.5604535341262817, 12: 1.5513306856155396, 311: 1.528236746788025, 38: 1.514960527420044, 11: 1.508102536201477, 269: 1.4972316026687622, 268: 1.4851301908493042, 271: 1.4836475849151611, 72: 1.4730554819107056, 303: 1.4667319059371948, 302: 1.4620106220245361, 132: 1.4454379081726074, 401: 1.413730263710022, 0: 1.4104417562484741, 386: 1.4076260328292847, 267: 1.384002447128296, 454: 1.3788859844207764, 37: 1.3706142902374268, 192: 1.3439428806304932, 433: 1.3050451278686523, 436: 1.2732532024383545, 215: 1.2578562498092651, 216: 1.249117374420166, 387: 1.245216965675354, 93: 1.1597611904144287, 322: 1.1582136154174805, 366: 1.1265974044799805, 385: 1.1165144443511963, 356: 1.1135244369506836, 427: 1.0405446290969849, 213: 1.003720760345459}
_KEY_IDX = [17, 84, 16, 314, 85, 315, 15, 181, 86, 14, 316, 87, 180, 317, 405, 179, 404, 178, 18, 402, 403, 83, 313, 91, 90, 200, 88, 89, 201, 199, 421, 320, 318, 321, 152, 175, 319, 182, 406, 377, 148, 208, 428, 171, 396, 95, 96, 77, 400, 146, 176, 194, 418, 324, 369, 140, 325, 32, 262, 307, 106, 375, 335, 378, 149, 395, 204, 170, 62, 78, 76, 424, 61, 431, 211, 379, 308, 292, 306, 150, 43, 291, 394, 273, 169, 183, 184, 430, 365, 185, 202, 210, 191, 422, 136, 364, 407, 408, 415, 409, 287, 57, 397, 135, 40, 74, 42, 288, 80, 172, 367, 212, 432, 434, 214, 270, 361, 310, 272, 304, 410, 138, 58, 186, 81, 435, 41, 13, 73, 39, 323, 82, 312, 416, 12, 311, 38, 11, 269, 268, 271, 72, 303, 302, 132, 401, 0, 386, 267, 454, 37, 192, 433, 436, 215, 216, 387, 93, 322, 366, 385, 356, 427, 213]
KEY_IDX = _KEY_IDX[:]
STABLE_IDX = [(196, 0.041146062314510345), (419, 0.041638121008872986), (174, 0.04262330010533333), (122, 0.045432206243276596), (188, 0.04641878977417946), (197, 0.048861853778362274), (399, 0.048883505165576935), (168, 0.04902322590351105), (236, 0.04929608106613159), (6, 0.0516219325363636), (3, 0.05169089883565903), (351, 0.05251950025558472), (456, 0.05543915927410126), (248, 0.056157588958740234), (412, 0.05668545514345169), (114, 0.05828186497092247), (195, 0.05984281748533249), (217, 0.061995554715394974), (343, 0.07085694372653961), (51, 0.07101800292730331)]
# LIP_IDX = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
OVAL_IDX = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
STATIC_OVAL_IDX = [356, 454, 323, 361, 288, 397, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
OUT_LIP_IDX = [181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91]
IN_LIP_IDX = [178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88]
LIP_IDX = OUT_LIP_IDX + IN_LIP_IDX

WIDE_BOUNDARY_IDX = [379, 365, 397, 435, 401, 352, 346, 347, 348, 349, 350, 277, 437, 420, 360, 278, 455, 305, 290, 328, 326, 2, 97, 99, 60, 166, 219, 48, 131, 198, 217, 114, 128, 121, 120, 119, 118, 117, 111, 116, 137, 177, 215, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378]
MASK_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 32, 36, 37, 38, 39, 40, 41, 42, 43, 47, 49, 50, 57, 59, 61, 62, 64, 72, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 98, 100, 101, 102, 106, 123, 126, 129, 135, 136, 138, 140, 142, 146, 147, 148, 149, 150, 152, 165, 167, 169, 170, 171, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 191, 192, 194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 235, 240, 262, 266, 267, 268, 269, 270, 271, 272, 273, 280, 287, 291, 292, 294, 302, 303, 304, 306, 307, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 327, 329, 330, 331, 335, 355, 358, 364, 365, 367, 369, 371, 375, 376, 377, 378, 379, 391, 393, 394, 395, 396, 400, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 415, 416, 418, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 436, 460]
WIDE_MASK_IDX = [0, 2, 11, 12, 13, 14, 15, 16, 17, 18, 32, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 57, 59, 60, 61, 62, 64, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99, 100, 101, 102, 106, 111, 114, 116, 117, 118, 119, 120, 121, 123, 126, 128, 129, 131, 135, 136, 137, 138, 140, 142, 146, 147, 148, 149, 150, 152, 164, 165, 166, 167, 169, 170, 171, 172, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 191, 192, 194, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 235, 240, 262, 266, 267, 268, 269, 270, 271, 272, 273, 277, 278, 279, 280, 287, 290, 291, 292, 294, 302, 303, 304, 305, 306, 307, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331, 335, 346, 347, 348, 349, 350, 352, 355, 358, 360, 364, 365, 367, 369, 371, 375, 376, 377, 378, 379, 391, 393, 394, 395, 396, 397, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 415, 416, 418, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 455, 460]
LEFT_EYE_IDX = [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382]
RIGHT_EYE_IDX = [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159]
_LEFT_EYE_IDX = [382, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]
_RIGHT_EYE_IDX = [159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158]
CONTOUR_IDX = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466]
ROI_IDX = LIP_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX
LEFT_EYEBROW_IDX = [336, 285, 295, 282, 283, 276, 300, 293, 334, 296]
RIGHT_EYEBROW_IDX = [70, 46, 53, 52, 65, 55, 107, 66, 105, 63]
LEFT_IRIS_IDX = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]
CHIN_IDX = [365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136]
OPENFACE_LEFT_EYEBROW_IDX = list(range(17, 22))
OPENFACE_RIGHT_EYEBROW_IDX = list(range(22, 27))
OPENFACE_LEFT_EYE_IDX = list(range(36, 42))
OPENFACE_RIGHT_EYE_IDX = list(range(42, 48))
OPENFACE_OUT_LIP_IDX = list(range(48, 61))
OPENFACE_IN_LIP_IDX = list(range(61, 68))
OPENFACE_NOSE_IDX = list(range(27, 36))


OPENFACE_EYE_IDX = list(range(17, 27)) + list(range(36, 48))
OPENFACE_LIP_IDX = list(range(48, 68))

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_file_list(data_dir, suffix=""):
    file_list = []

    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if suffix in filename:
                file_list.append(os.path.join(dirpath, filename))

    file_list = natsorted(file_list)

    return file_list

def mix_mesh_tensor(target, source):
    res = torch.tensor(source).to(source.device)
    res[KEY_IDX] = target[KEY_IDX]
    return res
 
def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
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


# Input :
#       reference(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of reference frame.
#       target(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of target frame.
# Output : 
#       R : 3x3 Rotation matrix(np.array)
#       c : scale value(float)
#       t : 3x1 translation matrix(np.array)


def Umeyama_algorithm(reference, target):
    # idx 2 -> nose, 130 -> left eye, 359 -> right eye
    idx_list = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10, 109, 108, 67, 69, 103, 104, 54, 68, 338, 337, 297, 299, 332, 333, 284, 298, 130, 243, 244, 359, 362, 463,
                21, 71, 162, 139, 156, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 301, 251, 55, 285, 193, 417, 122, 351, 196, 419, 3, 248, 51, 281,
                45, 275, 44, 274, 220, 440, 134, 363, 236, 456]
    # idx_list = [19, 243, 463]
    ref_points = []
    tgt_points = []

    for idx in idx_list:
        ref_points.append(reference[idx])
        tgt_points.append(target[idx])

    ref_points = np.array(ref_points)
    tgt_points = np.array(tgt_points)

    ref_mu = ref_points.mean(axis=0)
    tgt_mu = tgt_points.mean(axis=0)
    ref_var = ref_points.var(axis=0).sum()
    tgt_var = tgt_points.var(axis=0).sum()
    n, m = ref_points.shape
    covar = np.matmul((ref_points - ref_mu).T, tgt_points - tgt_mu) / n
    det_covar = np.linalg.det(covar)
    u, d, vh = np.linalg.svd(covar)
    detuv = np.linalg.det(u) * np.linalg.det(vh.T)
    cov_rank = np.linalg.matrix_rank(covar)
    S = np.identity(m)

    if cov_rank > m - 1:
        if det_covar < 0:
            S[m - 1, m - 1] = -1
    else: 
        if detuv < 0:
            S[m - 1, m - 1] = -1

    R = np.matmul(np.matmul(u, S), vh)
    c = (1 / tgt_var) * np.trace(np.matmul(np.diag(d), S))
    t = ref_mu.reshape(3, 1) - c * np.matmul(R, tgt_mu.reshape(3, 1))

    return R, t, c


def landmark_to_dict(landmark_list):
    landmark_dict = {}
    for idx, landmark in enumerate(landmark_list):
        landmark_dict[idx] = [landmark.x, landmark.y, landmark.z]

    return landmark_dict

def landmarkdict_to_normalized_mesh_tensor(landmark_dict):
    vertex_list = []
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        vertex_list.append(coord)
    
    if not ('R' in landmark_dict):
        return torch.tensor(vertex_list)
    
    R = torch.from_numpy(landmark_dict['R']).float()
    t = torch.from_numpy(landmark_dict['t']).float()
    c = float(landmark_dict['c'])
    vertices = torch.tensor(vertex_list).transpose(0, 1)
    norm_vertices = (c * torch.matmul(R, vertices) + t).transpose(0, 1)
    return norm_vertices

def normalize_mesh_tensor(landmark_tensor, R, t, c):
    vertices = landmark_tensor.transpose(0, 1)
    R = torch.from_numpy(landmark_dict['R']).float()
    t = torch.from_numpy(landmark_dict['t']).float()
    c = float(landmark_dict['c'])
    vertices = torch.tensor(vertex_list).transpose(0, 1)
    norm_vertices = (c * torch.matmul(R, vertices) + t).transpose(0, 1)
    return norm_vertices

def landmarkdict_to_mesh_tensor(landmark_dict):
    vertex_list = []
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        vertex_list.append(coord)

    vertices = torch.tensor(vertex_list)
    return vertices

def mesh_tensor_to_landmarkdict(mesh_tensor):
    landmark_dict = {}
    for i in range(mesh_tensor.shape[0]):
        landmark_dict[i] = mesh_tensor[i].tolist()
    
    return landmark_dict

def edge2color(edge_index):
    total = len(mp_face_mesh.FACEMESH_TESSELATION)
    id = (edge_index + 1) / total
    c = 127 + int(id * 128)
    return (c, c, c)

def draw_lips(keypoints, new_img, c = (255, 255, 255), th=1):
    keypoints = keypoints.astype(np.int32)
    for i in range(48, 59):
        cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
    for i in range(60, 67):
        cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)

def draw_mouth_mask(keypoints, shape):
    return draw_mask(keypoints[IN_LIP_IDX], shape)


def draw_mask(maskKp, shape, c=(255, 255, 255)):
  mask = np.zeros(shape, dtype=np.int32)
  center = np.mean(maskKp, axis=0)
  delta = maskKp - center
#   delta[1:8, 1] *= 2
  maskKp = center + delta
  _ = cv2.fillPoly(mask, [maskKp.astype(np.int32)], c)
  mask = mask.astype(np.float32) / 255.0
  return mask

# LEFT_EYE_IDX = []

def draw_section(sections, shape, section_config=[LEFT_EYEBROW_IDX, LEFT_EYE_IDX, LEFT_IRIS_IDX, RIGHT_EYEBROW_IDX, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, OUT_LIP_IDX, IN_LIP_IDX], groups=[0, 0, 1, 1, 2, 2], split=False, mask=None, C=(255, 255, 255)):
    # LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
    # RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
    # OUT_LIP_IDX = [48, 49, 50, 51, 52, 53, 54 ,55, 56, 57, 58, 59]
    # IN_LIP_IDX = [60, 61, 62, 63, 64, 65, 66, 67]
    # section_config = [LEFT_EYE_IDX, RIGHT_EYE_IDX, OUT_LIP_IDX, IN_LIP_IDX]
    # sections: np
    if mask is None:
        mask = np.zeros(shape, dtype=np.uint8)
        
    if len(sections) == 478:
        return get_mesh_image(torch.tensor(sections), shape)
    
    # united section
    # groups = [0] * len(groups)
    num_groups = len(set(groups))
    if split == True:
        masks = [mask.copy() for _ in range(num_groups)]
        
    for i, sec_idx in enumerate(section_config):
        if split:
            group = groups[i]
            sec_mask = masks[group]
        else:
            sec_mask = mask

        is_closed = True
        if len(sec_idx) == 2 and not sec_idx[1]:
            is_closed = False
            sec_idx = sec_idx[0]
            
        section = sections[:len(sec_idx)]
        sections = sections[len(sec_idx):]
        # print(f'section {i} - {sec_idx}: {section}')
        # if sec_idx == LEFT_EYE_IDX:
        #     reorder = list(map(lambda x: LEFT_EYE_IDX.index(x), _LEFT_EYE_IDX))
        #     section = section[reorder]
        # if sec_idx == RIGHT_EYE_IDX:
        #     reorder = list(map(lambda x: RIGHT_EYE_IDX.index(x), _RIGHT_EYE_IDX))
        #     section = section[reorder]
        
        # print(f'len sections: {len(sections)}')
        # section = np.array([
        #     [128, 128],
        #     [200, 200], 
        #     [0, 0]], np.int32)
        if len(section) == 1:
            _ = cv2.circle(sec_mask, section[0], 5, C, 1)
        else:
            _ = cv2.polylines(sec_mask, [section], is_closed, C, 1)
        
    # assert len(sections) == 0
    
    return mask if not split else masks

def get_seg(mesh_dict, shape):
    keypoints = np.array(list(mesh_dict.values())[:478])[:, :2]
    keypoints = keypoints.astype(np.int32)
    oval_idx = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
    out_lip_idx = [181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91]
    in_lip_idx = [178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88]
    maskKp = keypoints[oval_idx]
    seg = draw_mask(maskKp, shape)
    cv2.fillPoly(seg, [keypoints[out_lip_idx]], color=(2, 2, 2))
    cv2.fillPoly(seg, [keypoints[in_lip_idx]], color=(3, 3, 3))
    return seg

def get_lip_mask(mesh_dict, shape, boundary_idx=WIDE_BOUNDARY_IDX):
    keypoints = np.array(list(mesh_dict.values())[:478])[:, :2]
    keypoints = keypoints.astype(np.int32)[boundary_idx]
    lip_mask = draw_mask(keypoints, shape)
    return lip_mask

def get_mesh_image(mesh, frame_shape, mask_idx=None):
    mesh_dict = mesh_tensor_to_landmarkdict(mesh)
    image_rows, image_cols = frame_shape[1], frame_shape[0]
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)
    idx_to_coordinates = {}
    for idx, coord in mesh_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        x_px = min(math.floor(coord[0]), image_cols - 1)
        y_px = min(math.floor(coord[1]), image_rows - 1)
        landmark_px = (x_px, y_px)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    # get segment map
    segmap = get_seg(mesh_dict, (image_cols, image_rows, 3)) * 32

    # draw mesh
    connections = mp_face_mesh.FACEMESH_TESSELATION
    for edge_index, connection in enumerate(connections):
        start_idx = connection[0]
        end_idx = connection[1]
        if mask_idx is not None:
            if not start_idx in mask_idx or not end_idx in mask_idx:
                continue
        color = edge2color(edge_index)
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(segmap, 
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], 
                color,
                1
            )
    
    return segmap

def draw_mesh_image(mesh_dict, save_path, image_rows, image_cols, mask_idx=None):
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)
    idx_to_coordinates = {}
    for idx, coord in mesh_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        x_px = min(math.floor(coord[0]), image_cols - 1)
        y_px = min(math.floor(coord[1]), image_rows - 1)
        landmark_px = (x_px, y_px)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    # get segment map
    segmap = get_seg(mesh_dict, (image_cols, image_rows, 3)) * 32

    # draw mesh
    connections = mp_face_mesh.FACEMESH_TESSELATION
    for edge_index, connection in enumerate(connections):
        start_idx = connection[0]
        end_idx = connection[1]
        if mask_idx is not None:
            if not start_idx in mask_idx or not end_idx in mask_idx:
                continue
        color = edge2color(edge_index)
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(segmap, 
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], 
                color,
                1
            )
    cv2.imwrite(save_path, segmap)

def normalize_mesh(mesh_dict):
    mesh = landmarkdict_to_normalized_mesh_tensor(mesh_dict)
    return mesh_tensor_to_landmarkdict(mesh)

def normalize_meshes(mesh_dir, save_dir):
    mesh_filename_list = get_file_list(mesh_dir)
    os.makedirs(save_dir, exist_ok=True)
    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        mesh = landmarkdict_to_normalized_mesh_tensor(mesh_dict)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.pt')
        torch.save(mesh_tensor_to_landmarkdict(mesh), save_path)

def interpolate_zs_dir(data_dir, image_rows=256, image_cols=256):
    for mesh_dir in os.listdir(data_dir):
        if not mesh_dir.startswith('mesh_dict'):
            continue
        print(f"working on {mesh_dir}...")
        post_fix = mesh_dir[len('mesh_dict'):]
        save_dir = 'z' + post_fix
        interpolate_zs(os.path.join(data_dir, mesh_dir), os.path.join(data_dir, save_dir), image_rows, image_cols)


def interpolate_zs(mesh_dir, save_dir, image_rows, image_cols):
    mesh_filename_list = get_file_list(mesh_dir)
    os.makedirs(save_dir, exist_ok=True)
    # print(f'save dir: {save_dir}')
    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        mesh = landmarkdict_to_mesh_tensor(mesh_dict)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.pt')
        z = interpolate_z(mesh, image_rows, image_cols) # H x W x 1
        torch.save(z, save_path)

def interpolate_z(mesh, image_rows=256, image_cols=256):
    points, values = mesh[:, :2].numpy(), mesh[:, [2]].numpy()
    interp = LinearNDInterpolator(points, values, fill_value=0)
    X, Y = np.meshgrid(range(0, image_rows), range(0, image_cols))
    Z = interp(X, Y)
    return torch.from_numpy(Z) # image_rows x image_cols x 1

def interpolate_mesh_surface(mesh, value, image_rows=256, image_cols=256, fill_value=0):
    # mesh: N x 2
    # value: N x D
    mesh = mesh.numpy()
    value = value.numpy()
    interp = LinearNDInterpolator(mesh, value, fill_value=0)
    X, Y = np.meshgrid(range(0, image_rows), range(0, image_cols))
    Z = interp(X, Y)
    return torch.from_numpy(Z).float() # image_rows x image_cols x D
    
def draw_mesh_images(mesh_dir, save_dir, image_rows=256, image_cols=256):
    mesh_filename_list = get_file_list(mesh_dir)
    os.makedirs(save_dir, exist_ok=True)
    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.png')
        draw_mesh_image(mesh_dict, save_path, image_rows, image_cols)
    

def draw_mesh_images_dir(data_dir, image_rows=256, image_cols=256):
    for mesh_dir in os.listdir(data_dir):
        if not mesh_dir.startswith('mesh_dict'):
            continue
        if mesh_dir.endswith('.pt'):
            continue
        print(f"working on {mesh_dir}...")
        post_fix = mesh_dir[len('mesh_dict'):]
        save_dir = 'mesh_image' + post_fix
        draw_mesh_images(os.path.join(data_dir, mesh_dir), os.path.join(data_dir, save_dir), image_rows, image_cols)

def get_cov_max():
    cov = torch.load(os.path.join('/home/server25/minyeong_workspace/Lipsync3D/mp_cov.pt'))[:, LIP_IDX]
    cov_max = cov.max(1)[0]
    return cov_max

def construct_boundary(inner_set):
    # inner set: connected graph
    edges = mp_face_mesh.FACEMESH_TESSELATION
    boundary = set()
    for s, e in edges:
        if s in inner_set or e in inner_set:
                boundary.add(s)
                boundary.add(e)
    boundary = boundary - inner_set
    return boundary

def construct_max_boundary(initial_set, domain_set):
    # innder idx: connected set
    inner_set = initial_set.copy()
    edges = mp_face_mesh.FACEMESH_TESSELATION
    while True:
        print('inner set size: {}'.format(len(inner_set)))
        boundary = construct_boundary(inner_set)
        new_member = boundary & domain_set
        if len(new_member) == 0:
            break
        inner_set.update(new_member)
    return boundary, inner_set

def split_groups(dummy):
    # dummy: set
    edges = mp_face_mesh.FACEMESH_TESSELATION
    lines = []
    for e in dummy:
        comp = []
        for s in lines:
            for e0 in s:
                if (e0, e) in edges or (e, e0) in edges:
                    comp.append(s)
                    break
        new_set = set([e])
        for s in comp:
            lines.remove(s)
            new_set.update(s)
        lines.append(new_set)
    return lines

def reorder_ring_in_order(points):
    edges = mp_face_mesh.FACEMESH_TESSELATION
    points = list(points)
    line = [points.pop()]
    while len(points) > 0 :
        s = line[-1]
        for p in points:
            if (s, p) in edges or (p, s) in edges:
                line.append(p)
                points.remove(p)
                break
        if s == line[-1]:
            line.pop()
    return line

def construct_stack(data_dir, include_audio=True):
    mesh_stack = []
    vid_path = data_dir
    dict_dir = os.path.join(vid_path, 'mesh_dict_normalized')
    dicts = os.listdir(dict_dir)
    dicts.sort()
    dicts = dicts[:-5]
    for d in dicts:
        landmark_dict = torch.load(os.path.join(dict_dir, d))
        mesh = landmarkdict_to_mesh_tensor(landmark_dict)
        mesh_stack.append(mesh)
        if include_audio:
            with open(os.path.join(vid_path, 'audio', '{:05d}.pickle'.format(int(d[:-3]) - 1)), 'rb') as f:
                audio = pkl.load(f)
            audio_list.append(audio)
    mesh_stack = torch.stack(mesh_stack, dim=0)
    return (mesh_stack, audio_list) if include_audio else mesh_stack

def construct_stack_dir(data_dir, vid_list_name='video_list.pt', include_audio=True):
    mesh_stack = []
    if include_audio:
        audio_list = []
    vid_list = torch.load(os.path.join(data_dir, vid_list_name))
    for vid_name in vid_list:
            local_mesh_stack = []
            if include_audio:
                local_audio_stack = []
            vid_path = os.path.join(data_dir, vid_name)
            dict_dir = os.path.join(vid_path, 'mesh_dict_normalized')
            dicts = os.listdir(dict_dir)
            dicts.sort()
            dicts = dicts[:-5]
            for d in dicts:
                landmark_dict = torch.load(os.path.join(dict_dir, d))
                mesh = landmarkdict_to_mesh_tensor(landmark_dict)
                local_mesh_stack.append(mesh)
                if include_audio:
                    with open(os.path.join(vid_path, 'audio', '{:05d}.pickle'.format(int(d[:-3]) - 1)), 'rb') as f:
                        audio = pkl.load(f)
                    local_audio_list.append(audio)
                    audio_list += local_audio_list
            torch.save(torch.stack(local_mesh_stack, dim=0), os.path.join(vid_path, 'mesh_stack.pt'))
            mesh_stack += local_mesh_stack
    mesh_stack = torch.stack(mesh_stack, dim=0)
    return (mesh_stack, audio_list) if include_audio else mesh_stack

def project_mesh(data_dir, pca_path=None, ref_name='mesh_dict_reference.pt'):
    mesh_path = os.path.join(data_dir, 'mesh_stack.pt')
    reference_mesh_path = os.path.join(data_dir, ref_name)
    mesh_stack = torch.load(mesh_path)  # L x N0 x 3
    reference_mesh = landmarkdict_to_mesh_tensor(torch.load(reference_mesh_path))    # N0 x 3
    mesh_stack_centered = mesh_stack - reference_mesh[None] # L x N0 x 3
    roi = mesh_stack_centered[:, LIP_IDX] # L x N x 3
    roi = roi.flatten(-2)   # L x N * 3
    roi_scaled = roi / 128
    if pca_path is None:
        pca = torch.pca_lowrank(roi_scaled, q=20)
        return pca
    else:
        _u, _s, _v = torch.load(pca_path)   # _u: _ x q, _s: q, _v: d x q
        u = roi @ _v @ torch.diag(_s).inverse()
        return u, _s, _v
    
def construct_pool(data_dir, N=None, pool_name='pool'):
    mesh_path = os.path.join(data_dir, 'mesh_stack.pt')
    mesh_stack = torch.load(mesh_path)
    N0 = len(mesh_stack)
    if N is None:
        N = N0
    deck = list(range(N0 - N + 1))
    random.shuffle(deck)
    pool_start_idx = random.choice(deck)
    pool_idx = list(range(pool_start_idx, pool_start_idx + N))
    mesh_pool = mesh_stack[pool_idx]
    pool_dir = os.path.join(data_dir, pool_name)
    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(os.path.join(pool_dir, 'audio'), exist_ok=True)
    torch.save(mesh_pool, os.path.join(pool_dir, 'mesh_stack.pt'))
    torch.save(pool_idx, os.path.join(pool_dir, 'idx.pt'))
    mesh_pca_path = os.path.join(data_dir, 'mesh_pca.pt')
    _pca = torch.load(mesh_pca_path)
    _coef = _pca[0]
    coef = _coef[pool_idx]
    pca = (coef, _pca[1], _pca[2])
    torch.save(pca, os.path.join(pool_dir, 'mesh_pca.pt'))
    audio_list = os.listdir(os.path.join(data_dir, 'audio'))
    audio_list.sort()
    for i, idx in enumerate(pool_idx):
        audio_path = os.path.join(data_dir, 'audio', audio_list[idx])
        with open(audio_path, 'rb') as f:
            audio = pkl.load(f)
        with open(os.path.join(pool_dir, 'audio', '{:05d}.pickle'.format(i)), 'wb') as f:
            pkl.dump(audio, f)
            

def pca_mesh(mesh_stack_path, save_name, pca_path=None):
    mesh_stack = torch.load(mesh_stack_path) / 128 - 1    # B x N x 3
    if pca_path is not None:
        pca = torch.load(pca_path)
        mesh_mean = pca['mean']
        _u, s, v = pca['pca']
        expression_stack = mesh_stack - mesh_mean[None]  # B x N * 3
        u = expression_stack.flatten(1) @ v @torch.diag(s).inverse()   # B x q
        res = u, s, v
    else:
        mesh_mean = mesh_stack.mean(dim=0)   # N x 3
        expression_stack = mesh_stack - mesh_mean[None] # B x N * 3
        res = torch.pca_lowrank(expression_stack.flatten(1), q=80)

    torch.save({'mean': mesh_mean.view(-1, 3), 'pca': res}, save_name)


def matrix2euler(r, seq='xyz', degrees=False):
    # R(numpy): 3 x 3
    return R.from_matrix(r).as_euler(seq=seq, degrees=degrees)

def euler2matrix(angles, seq='xyz', degrees=False):
    # angles = 3-d array
    return R.from_euler(seq, angles, degrees).as_matrix()

def block_diagonal_batch(A, N):
    # A: B x d x d
    # out: B x N * d x N * d block diagonal
    B, d = A.shape[:2]
    eyes = torch.eye(N * d).to(A.device)
    eyes = eyes.reshape(N, d, N, d).permute(0, 2, 1, 3).reshape(N * N, d, d)
    out = torch.einsum('bij,njk->bnik', A, eyes) # B x N ** 2 x d x d
    out = out.reshape(B, N, N, d, d).permute(0, 1, 3, 2, 4).reshape(B, N * d, N * d)
    return out