# import dlib
import numpy as np
from collections import OrderedDict
import cv2
import eos
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import face_alignment
import os
import mediapipe as mp
from utils.ffhq_align import image_align
from utils.util import euler2matrix, matrix2euler
import imageio
from tqdm import tqdm
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
from PIL import Image
import shutil

# from batch_face import RetinaFace, LandmarkPredictor

def multiplyABC(A, B, C):
    temp = np.dot(A, B)
    return np.dot(temp, C)

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def construct_command(query):
    inp = query['inp']
    start = query['start']
    time = query['time']
    w = query['w']
    h = query['h']
    left = query['left']
    top = query['top']
    scale = query['scale']
    output = query['output']
    fps = query['fps']

    return f'ffmpeg -y -i {inp} -ss {start} -t {time} -filter:v "fps={fps}, crop={w}:{h}:{left}:{top}, scale={scale}" {output}'

def compute_bbox(start, end, fps, tube_bbox, frame_shape, inp, image_shape, output, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'

    query = {'inp': inp, 'start': start, 'time': time, 'w': w, 'h': h, 'left': left, 'top': top, 'scale': scale, 'output': output, 'fps': fps}
    
    return query


def compute_bbox_trajectories(trajectories, fps, frame_shape, min_frames, inp, image_shape, increase, output):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > min_frames:
            command = compute_bbox(start, end, fps, tube_bbox, frame_shape, inp=inp, image_shape=image_shape, increase_area=increase, output=output)
            commands.append(command)
    return commands


class LandmarkModel():
    def __init__(self, config_dir, gpu=[0]):
        self.config_dir = config_dir
        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])

        # print(f"directory {os.listdir('.')}")
        self.gpu = gpu
        self.model = eos.morphablemodel.load_model(f"{self.config_dir}/sfm_shape_3448.bin")
        self.blendshapes = eos.morphablemodel.load_blendshapes(f"{self.config_dir}/expression_blendshapes_3448.bin")
        self.morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(self.model.get_shape_model(), self.blendshapes,
                                                                                color_model=eos.morphablemodel.PcaModel(),
                                                                                vertex_definitions=None,
                                                                                texture_coordinates=self.model.get_texture_coordinates())
        self.landmark_mapper = eos.core.LandmarkMapper(f'{self.config_dir}/ibug_to_sfm.txt')
        self.edge_topology = eos.morphablemodel.load_edge_topology(f'{self.config_dir}/sfm_3448_edge_topology.json')
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(f'{self.config_dir}/ibug_to_sfm.txt')
        self.model_contour = eos.fitting.ModelContour.load(f'{self.config_dir}/sfm_model_contours.json')
        self.landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=f'cuda:{self.gpu[0]}')
        # self.detector = RetinaFace(0)
        # self.predictor = LandmarkPredictor(0)

        # initialize mediapipe
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def preprocess_image(self, inp, output):
        landmarks_detector = self.fa
        frame = imageio.imread(inp)
        frame = frame[:, :, :3]

        if max(frame.shape[0], frame.shape[1]) > 640:
            scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
            frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
            frame = img_as_ubyte(frame)
        else:
            scale_factor = 1
            frame = img_as_ubyte(frame)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(frame), start=1):
            aligned_face_path = output
            result_img = np.array(image_align(inp, face_landmarks * scale_factor))[:, :, :3]
            
            # ## segmentation
            # seg_results = self.selfie_segmentation.process(result_img)
            # seg_mask = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.5
            # print(f'seg mask: {seg_mask}')
            # bg = np.ones_like(seg_mask) * 239
            # result_img = np.where(seg_mask, img_as_ubyte(result_img), bg.astype(np.uint8))
        
            Image.fromarray(result_img).save(aligned_face_path, 'PNG')

            break

    def preprocess_video(self, inp, output, image_shape=(256, 256), increase=0.1, iou_with_initial=0.25, min_frames=1):
        # if os.path.exists(output):
        #     return output
        
        tmp_dir = os.path.join(os.path.dirname(output), 'preprocess_video_tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            
        frames_dir = os.path.join(os.path.dirname(output), 'frames')
        # if os.path.exists(frames_dir):
        #     shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        
        
        fa = self.fa
        video = imageio.get_reader(inp)

        trajectories = []
        previous_frame = None
        fps = video.get_meta_data()['fps']
        print(f'preprocessing video of fps={fps}')
        commands = []
        try:
            for i, frame in tqdm(enumerate(video)):
                frame_shape = frame.shape
                frame = frame[:, :, :3]
                bboxes =  extract_bbox(frame, fa)
                ## For each trajectory check the criterion
                not_valid_trajectories = []
                valid_trajectories = []

                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    intersection = 0
                    for bbox in bboxes:
                        intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                    if intersection >  iou_with_initial:
                        valid_trajectories.append(trajectory)
                    else:
                        not_valid_trajectories.append(trajectory)

                commands += compute_bbox_trajectories(not_valid_trajectories, fps, frame_shape, min_frames, inp, image_shape, increase, output)
                trajectories = valid_trajectories

                ## Assign bbox to trajectories, create new trajectories
                for bbox in bboxes:
                    intersection = 0
                    current_trajectory = None
                    for trajectory in trajectories:
                        tube_bbox = trajectory[0]
                        current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                        if intersection < current_intersection and current_intersection > iou_with_initial:
                            intersection = bb_intersection_over_union(tube_bbox, bbox)
                            current_trajectory = trajectory

                    ## Create new trajectory
                    if current_trajectory is None:
                        trajectories.append([bbox, bbox, i, i])
                    else:
                        current_trajectory[3] = i
                        current_trajectory[1] = join(current_trajectory[1], bbox)


        except Exception as e:
            print (e)

        commands += compute_bbox_trajectories(trajectories, fps, frame_shape, min_frames, inp, image_shape, increase, output)
        
        cmd_strings = []
        for i, cmd in enumerate(commands):
            cmd['output'] = os.path.join(tmp_dir, '{:05d}.mp4'.format(i))
            cmd_string = construct_command(cmd)
            cmd_strings.append(cmd_string)
            os.system(cmd_string)
        print(f'cmd strings: {cmd_strings}')

        concat_cmd = 'concat:' + '|'.join([os.path.join(tmp_dir, '{:05d}.mp4'.format(i)) for i in range(len(cmd_strings))])
        os.system(f'ffmpeg -y -i "{concat_cmd}" -c copy {output}')


        return output

    def get_landmarks_fa(self, frame):
        frame = frame[:, :, :3]
        H, W = frame.shape[0:2]
        bb = (0, 0, H - 1, W - 1)
        if max(H, W) > 640:
            scale_factor =  max(H, W) / 640.0
            frame = resize(frame, (int(H / scale_factor), int(W / scale_factor)))
            frame = img_as_ubyte(frame)
        else:
            scale_factor = 1
            frame = img_as_ubyte(frame)
        # faces = self.detector([frame])
        # f_boxes = [np.array(bb)[None]]
        # print(f'faces: {faces}')
        # frame_landmarks = self.predictor(f_boxes, [frame])
        # print(f'frame_landmarks - type: {type(frame_landmarks)}')
        # print(f'frame_landmarks - shape: {frame_landmarks.shape}')
        # print(f'frame_landmarks - data: {frame_landmarks}')
        frame_landmarks = self.fa.get_landmarks(frame)[0].astype(int) * scale_factor
        # print(f'delta: {frame_landmarks[0][0].astype(int) - GT_frame_landmarks}')
        frame_landmarks = np.stack([frame_landmarks[:, 0].clip(0, H - 1), frame_landmarks[:, 1].clip(0, W - 1)], axis=1)
        return (bb, frame_landmarks)

    def get_landmarks_batch(self, frames):
        frames = np.array(frames)
        frames = frames[:, :3]
        H, W = frames.shape[2:4]
        bb = (0, 0, H - 1, W - 1)
        batch_landmarks = []
        for i, frame in enumerate(frames):
            frame = frame.transpose((1, 2, 0))
            if max(H, W) > 640:
                scale_factor =  max(H, W) / 640.0
                frame = resize(frame, (int(H / scale_factor), int(W / scale_factor)))
                frame = img_as_ubyte(frame)
            else:
                scale_factor = 1
                frame = img_as_ubyte(frame)
            # faces = self.detector([frame])
            # f_boxes = [np.array(bb)[None]]
            # print(f'faces: {faces}')
            # frame_landmarks = self.predictor(f_boxes, [frame])
            # print(f'frame_landmarks - type: {type(frame_landmarks)}')
            # print(f'frame_landmarks - shape: {frame_landmarks.shape}')
            # print(f'frame_landmarks - data: {frame_landmarks}')
            frame_landmarks = self.fa.get_landmarks(frame)[0].astype(int) * scale_factor
            # print(f'delta: {frame_landmarks[0][0].astype(int) - GT_frame_landmarks}')
            frame_landmarks = np.stack([frame_landmarks[:, 0].clip(0, H - 1), frame_landmarks[:, 1].clip(0, W - 1)], axis=1)
            batch_landmarks.append(frame_landmarks)
        batch_landmarks = np.array(batch_landmarks)
        
        return batch_landmarks

    def normalize_mesh(self, landmarks, image_height, image_width, z_mean=None, R_noise=None, t_noise=None):
        eos_landmarks = []
        # print(f'landmarks: {landmarks}')
        # print(f'image size: {image_height}, {image_width}')
        for idx in range(0,68):
            eos_landmarks.append(eos.core.Landmark(str(idx+1), [float(landmarks[idx,0]), float(landmarks[idx,1])]))
        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self.morphablemodel_with_expressions,
            eos_landmarks, self.landmark_mapper, image_width, image_height, self.edge_topology, self.contour_landmarks, self.model_contour)

        vertices = np.array(mesh.vertices)
        vertices = np.append(vertices, np.ones((vertices.shape[0], 1)), 1)

        w2, h2 = image_width/2, image_height/2
        viewport = np.array([[w2, 0, 0, w2],
                            [0, -h2, 0, h2],
                            [0, 0, 1, -75],
                            [0, 0, 0, 1]])
        proj = pose.get_projection()
        view = pose.get_modelview()
        # if R_noise is not None:
        #     noise_euler = R_noise * np.pi * (2 * np.random.rand(3) - 1)
        #     noise_matrix = euler2matrix(noise_euler)
        #     view[:3, :3] = noise_matrix @ view[:3, :3]
        # if t_noise is not None:
        #     viewport[:3, 3] += t_noise * (2 * np.random.rand(3) - 1)
            
        if R_noise:
            noise_euler = R_noise * np.pi * (2 * np.random.rand(3) - 1)
            noise_matrix = euler2matrix(noise_euler)
            view[:3, :3] = noise_matrix @ view[:3, :3]
        
        # if t_noise is not None:
        #     viewport[:3, 3] += t_noise * np.random.randn(3)
        
        # # print(f'view: {view}')
        # print(f'view rot: {matrix2euler(view[:3, :3])}')
        # while True:
        #     continue
        proj[3, 3] = 1
        a = multiplyABC(viewport, proj, view)
        
        a = a.transpose()
        mesh_3d_points = np.dot(vertices, a)
        
        if z_mean is not None:
            z_bias = mesh_3d_points[:, 2].mean()
            viewport[2, 3] = viewport[2, 3] - z_bias
            # viewport = np.array([[w2, 0, 0, w2],
            #                     [0, -h2, 0, h2],
            #                     [0, 0, 1, -75 - z_bias],
            #                     [0, 0, 0, 1]])
            a = multiplyABC(viewport, proj, view)
            
            a = a.transpose()
            mesh_3d_points = np.dot(vertices, a)
            # print(f'z_mean adjusted: {z_bias}')
        
        # print(f'vertices: {vertices}')
        # print()
        # landmark index in mesh
        # Ind = np.zeros((68,))
        Ind = np.array([2127, 2508, 2076, 2257, 2083, 1767,  945, 2505,   33, 2237,  946,
        1773, 1886,  900, 1890, 1749,  776,  225,  229,  233, 2086,  157,
            590, 2091,  666,  662,  658, 2842,  379,  272,  114,  100, 2794,
            270, 2797,  537,  177,  172,  191,  181,  173,  174,  614,  624,
            605,  610,  607,  606,  398,  315,  413,  329,  825,  736,  812,
            841,  693,  411,  264,  431, 3253,  416,  423,  828,  821,  817,
            442,  404]).astype(int)
        # ibug2sfm = {'45': '605', '30': '272', '44': '624', '39': '191', '57': '693', '38': '172', '53': '825', '43': '614', '63': '423', '36': '537', '35': '2797', '27': '658', '34': '270', '33': '2794', '31': '114', '55': '812', '42': '174', '49': '398', '41': '173', '28': '2842', '18': '225', '46': '610', '48': '606', '52': '329', '59': '264', '60': '431', '51': '413', '50': '315', '54': '736', '62': '416', '47': '607', '58': '411', '64': '828', '68': '404', '56': '841', '29': '379', '40': '181', '25': '666', '22': '157', '9': '33', '67': '442', '24': '2091', '23': '590', '21': '2086', '20': '233', '32': '100', '26': '662', '37': '177', '66': '817', '19': '229'}
        # Ind = 
        # for (i, (x, y)) in enumerate(landmarks):
        #     # k = str(i + 1)
        #     # if k in ibug2sfm:
        #     #     Ind[i] = int(ibug2sfm[k])
        #     #     print(f'{i}-th found in dictionary: ')
        #     #     continue
        #     d = (np.square(x - w2 - mesh_3d_points[:, 0]) + np.square(y - h2 - mesh_3d_points[:,1]))
        #     # print(f'd shape, value: {d.shape}, {d}')
        #     Ind[i] = np.argmin(d)
        #     print(f'found id with d = {d[int(Ind[i])]}')
        rotation_angle = pose.get_rotation_euler_angles()


        landmarks_3d = np.concatenate([landmarks[:, [0]], landmarks[:, [1]], mesh_3d_points[Ind, 2:3]], axis=1)
        # landmarks_3d = mesh_3d_points[Ind]
        # print(f'mesh 3d points: {landmarks_3d}')
        # print(f'view port: {viewport}')
        # print(f'projection: {proj}')
        # print(f'view: {view}')
        # print(f'fourth column: {mesh_3d_points[Ind, 3]}')
        normalizer = lambda x_3d:  (((np.concatenate([x_3d, np.ones((len(x_3d), 1))], axis=1) @ np.linalg.inv(viewport.T))[:, :3] + np.array([[1, 1, 0]])) @ np.linalg.inv(proj[:3, :3].T) - np.ones((1, 3)) * view[:3, -1:].T) @ np.linalg.inv(view.T[:3, :3])
        # normalized_landmarks_3d = (((landmarks_3d @ np.linalg.inv(viewport.T))[:, :3] + np.array([[1, 1, 0]])) @ np.linalg.inv(proj[:3, :3].T) - np.ones((1, 3)) * view[:3, -1:].T) @ np.linalg.inv(view.T[:3, :3])
        normalized_landmarks_3d = normalizer(landmarks_3d)

        # print(f'normed mesh: {mesh_3d_points}')
        normalized_landmarks_3d = normalized_landmarks_3d[:, :3]
        # landmarks_3d = vertices[Ind, :3]
        # print(f'a shape: {a}')
        # print(f'value: {vertices}')
        # print(f'proj: {pose.get_projection()}')
        # print(f'view: {pose.get_modelview()}')
        # while True:
        #     continue
        return normalized_landmarks_3d, {'U': a, 'viewport': viewport, 'proj': proj, 'view': view, 'normalizer': normalizer, 'landmarks_3d': landmarks_3d}, Ind
        