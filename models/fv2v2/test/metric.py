"""Code for computation of PLCC, SRCC and KRCC between
    PIQ metrics predictions and ground truth scores from MOS databases.
"""
from tkinter.tix import Y_REGION
import piq
import tqdm
import torch
import torch.nn.functional as F
import argparse
import functools
import torchvision

import pandas as pd
import numpy as np
from scipy import stats

from typing import List, Callable, Tuple
from pathlib import Path
from skimage.io import imread
from skimage.util import img_as_float32
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from torch import nn
from itertools import chain
import os
from modules.landmark_model import LandmarkModel

# L1 ↓ FID ↓ SSIM ↑ MS-SSIM ↑ LPIPS ↓ PSNR ↓ AKD ↓ 


## load images from directory
class DATASET(Dataset):
    def __init__(self, path):
        self.path = path
        self.frames = os.listdir(self.path)
        
    def get_path(self, index):
        path = os.path.join([self.path, self.frames[index]])
        return path
    
    def __getitem__(self, index):
        frame = img_as_float32(imread(self.get_path(index)))
        return frame
    
    def __len__(self):
        return len(self.frames)
    
class MetricEvaluater():
    def __init__(self, config, landmark_model=None):
        self.config = config
        self.dataset = DATASET
        self.dataloader = DataLoader
        self.landmark_model = LandmarkModel(config.lanmarkmodel_path) if landmark_model is None else landmark_model
        
    def get_dataloader(self, path, **kwargs):
        dataset = self.dataset(path)
        return self.dataloader(dataset)
    
    def get_paired_frames(self, x, y):
        # x, y: image directory
        files_x = os.listdir(x)
        files_y = os.listdir(y)

        pairs = []
        for file in files_x:
            if file in files_y:
                path_x = os.path.join(x, file)
                path_y = os.path.join(y, file)
                frame_x = img_as_float32(imread(path_x))
                frame_y = img_as_float32(imread(path_y))
                pairs.append([frame_x, frame_y])
        
        pairs = torch.tensor(np.array(pairs)).permute(0, 3, 1, 2)
        
        return pairs
    
    def get_frames(self, x):
        # x, y: image directory
        files_x = os.listdir(x)

        frames = []
        
        for file in files_x:
            path_x = os.path.join(x, file)
            frame_x = img_as_float32(imread(path_x))
            frames.append(frame_x)
        
        frames = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)
        
        return frames
    
    def get_frame(self, path):
        frame = img_as_float32(imread(path))
        frame = torch.tensor(frame).unsqueeze(0).permute(0, 3, 1, 2)
        
        return frame
    
    def L1(self, x, y):
        # x: generated, y: ground truth tensor of (B x channel x H x W)
        return torch.abs((x - y)).mean()
    
    def FID(self, x, y, is_path=True):
        metric = piq.fid()
        if is_path:
            dl_x = self.get_dataloader(x)
            dl_y = self.get_dataloader(y)
            x = metric.compute_feats(dl_x)
            y = metric.compute_feats(dl_y)
        return metric(x, y)


    def SSIM(self, x, y, is_path=True):
        metric = piq.ssim()
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)

    def MS_SSIM(self, x, y, is_path=True):
        metric = piq.multi_scale_ssim()
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)
    
    def LPIPS(self, x, y, is_path=True):
        metric = piq.LPIPS()
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)
    
    def PSNR(self, x, y, is_path=True):
        metric = piq.psnr()
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        return metric(x, y)
    
    
    def AKD(self, x, y, is_path):
        # x,y : B x C x H x W images
        if is_path:
            pairs = self.get_paired_frames(x, y)
            x = pairs[:, 0]
            y = pairs[:, 1]
        bs = len(lm_x)
        lm_x = self.landmark_model.get_landmarks_batch(x).view(bs, -1)
        lm_y = self.landmark_model.get_landmarks_batch(y).view(bs, -1)
        
        res = np.linalg.norm(lm_x - lm_y, axis=-1).mean()
        return res
    
    def calc_dist_uniformity(self, x, func=None, is_path=True):
        if is_path:
            # x: path to samples in .txt format
            samples = np.loadtxt(x)
        else:
            smaples = x
        if func is not None:
            samples = func(samples)
        return stats.kstest(samples, stats.uniform.cdf)
    
    def calc_dist_similarity(self, x, y, is_path=True):
        pass
    
    def CSIM(self, x, y, is_path):
        # x, y: B x C x H x W
        if is_path:
            x = self.get_frames(x)
            y = self.get_frame(y)
        
        # extract feature using Inception / VGG
        # x_feat = 
        # y_feat = 
        
        res = torch.einsum('bd,cd->bc', F.normalize(x_feat), F.normalize(y_feat))
            
        return res
    
    def AUCON(self, x, y, is_path):
        pass
        
    def run(self, metrics=[]):
        pass

@dataclass
class Metric:
    name: str
    functor: Callable
    category: str  # FR - full-reference, NR - no-reference, DB - distribution-based

    def __post_init__(self):
        valid_categories = {'FR', 'NR', 'DB'}
        assert self.category in valid_categories, f'Provided category [{self.category}] is invalid. ' \
                                                  f'Provide one of: {valid_categories}'


torch.multiprocessing.set_sharing_strategy('file_system')

METRICS = {
    # Full-reference
    "PSNR": Metric(name="PSNR", functor=functools.partial(piq.psnr, data_range=255., reduction='none'), category='FR'),
    "SSIM": Metric(name="SSIM", functor=functools.partial(piq.ssim, data_range=255., reduction='none'), category='FR'),
    "MS-SSIM": Metric(name="MS-SSIM", functor=functools.partial(piq.multi_scale_ssim, data_range=255.,
                                                                reduction='none'), category='FR'),
    "IW-SSIM": Metric(name="IW-SSIM", functor=functools.partial(piq.information_weighted_ssim, data_range=255.,
                                                                reduction='none'), category='FR'),
    "VIFp": Metric(name="VIFp", functor=functools.partial(piq.vif_p, data_range=255., reduction='none'), category='FR'),
    "GMSD": Metric(name="GMSD", functor=functools.partial(piq.gmsd, data_range=255., reduction='none'), category='FR'),
    "MS-GMSD": Metric(name="MS-GMSD", functor=functools.partial(piq.multi_scale_gmsd, data_range=255.,
                                                                reduction='none'), category='FR'),
    "MS-GMSDc": Metric(name="MS-GMSDc", functor=functools.partial(piq.multi_scale_gmsd, data_range=255.,
                                                                  chromatic=True, reduction='none'), category='FR'),
    "FSIM": Metric(name="FSIM", functor=functools.partial(piq.fsim, data_range=255.,
                                                          chromatic=False, reduction='none'), category='FR'),
    "FSIMc": Metric(name="FSIMc", functor=functools.partial(piq.fsim, data_range=255.,
                                                            chromatic=True, reduction='none'), category='FR'),
    "VSI": Metric(name="VSI", functor=functools.partial(piq.vsi, data_range=255., reduction='none'), category='FR'),
    "SR-SIM": Metric(name="SR-SIM", functor=functools.partial(piq.srsim, data_range=255.,
                                                              reduction='none'), category='FR'),
    "SR-SIMc": Metric(name="SR-SIM", functor=functools.partial(piq.srsim, data_range=255.,
                                                               reduction='none', chromatic=True), category='FR'),
    "HaarPSI": Metric(name="HaarPSI", functor=functools.partial(piq.haarpsi, data_range=255.,
                                                                reduction='none'), category='FR'),
    "MDSI": Metric(name="MDSI", functor=functools.partial(piq.mdsi, data_range=255., reduction='none'), category='FR'),
    "LPIPS-vgg": Metric(name="LPIPS-vgg", functor=piq.LPIPS(replace_pooling=False, reduction='none'), category='FR'),
    "DISTS": Metric(name="DISTS", functor=piq.DISTS(reduction='none'), category='FR'),
    "PieAPP": Metric(name="PieAPP", functor=piq.PieAPP(data_range=255., reduction='none'), category='FR'),
    "Content": Metric(name="Content", functor=piq.ContentLoss(reduction='none'), category='FR'),
    "Style": Metric(name="Style", functor=piq.StyleLoss(reduction='none'), category='FR'),
    "DSS": Metric(name="DSS", functor=functools.partial(piq.dss, data_range=255., reduction='none'), category='FR'),

    # No-reference
    "BRISQUE": Metric(name="BRISQUE", functor=functools.partial(piq.brisque, data_range=255., reduction='none'),
                      category='NR'),

    # Distribution-based
    "IS": Metric(name="IS", functor=piq.IS(distance='l1'), category='DB'),
    "FID": Metric(name="FID", functor=piq.FID(), category='DB'),
    "GS": Metric(name="GS", functor=piq.GS(), category='DB'),
    "KID": Metric(name="KID", functor=piq.KID(), category='DB'),
    "MSID": Metric(name="MSID", functor=piq.MSID(), category='DB'),
    "PR": Metric(name="PR", functor=piq.PR(), category='DB')
}

METRIC_CATEGORIES = {cat: [k for k, v in METRICS.items() if v.category == cat] for cat in ['FR', 'NR', 'DB']}


class TID2013(Dataset):
    r""" A class to evaluate on the KADID10k dataset.
    Note that the class is callable. The values are returned as a result of calling the __getitem__ method.

    Args:
        root: Root directory path.
    Returns:
        x: image with some kind of distortion in [0, 1] range
        y: image without distortion in [0, 1] range
        score: MOS score for this pair of images
    """
    _filename = "mos_with_names.txt"

    def __init__(self, root: Path = "datasets/tid2013") -> None:
        assert root.exists(), \
            "You need to download TID2013 dataset first. Check http://www.ponomarenko.info/tid2013"

        df = pd.read_csv(
            root / self._filename,
            sep=' ',
            names=['score', 'dist_img'],
            header=None
        )
        df["ref_img"] = df["dist_img"].apply(lambda x: f"reference_images/{(x[:3] + x[-4:]).upper()}")
        df["dist_img"] = df["dist_img"].apply(lambda x: f"distorted_images/{x}")

        self.scores = df['score'].to_numpy()
        self.df = df[["dist_img", 'ref_img', 'score']]
        self.root = root

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_path = self.root / self.df.iloc[index][0]
        y_path = self.root / self.df.iloc[index][1]
        score = self.scores[index]

        # Load image and ref, convert to tensor.
        x = torch.tensor(imread(x_path), dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(imread(y_path), dtype=torch.float32).permute(2, 0, 1)

        return x, y, score

    def __len__(self) -> int:
        return len(self.df)


class KADID10k(TID2013):
    r""" A class to evaluate on the KADID10k dataset.
    One can get the dataset via the direct link: https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip.
    Note that the class is callable. The values are returned as a result of calling the __getitem__ method.

    Args:
        root: Root directory path.
    Returns:
        x: image with some kind of distortion in [0, 1] range
        y: image without distortion in [0, 1] range
        score: MOS score for this pair of images
    """
    _filename = "dmos.csv"

    def __init__(self, root: Path = "datasets/kadid10k") -> None:
        assert root.exists(), \
            "You need to download KADID10K dataset first. " \
            "Check http://database.mmsp-kn.de/kadid-10k-database.html " \
            "or download via the direct link https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip"

        # Read file mith DMOS
        self.df = pd.read_csv(root / self._filename)
        self.df.rename(columns={"dmos": "score"}, inplace=True)
        self.scores = self.df["score"].to_numpy()
        self.df = self.df[["dist_img", 'ref_img', 'score']]

        self.root = root / "images"


class PIPAL(TID2013):
    r""" A class to evaluate on the train set of the PIPAL dataset.
    Note that the class is callable. The values are returned as a result of calling the __getitem__ method.

    Args:
        root: Root directory path.
    Returns:
        x: image with some kind of distortion in [0, 1] range
        y: image without distortion in [0, 1] range
        score: MOS score for this pair of images
    """

    def __init__(self, root: Path = Path("data/raw/pipal")) -> None:
        assert root.exists(), \
            "You need to download PIPAL dataset. Check https://www.jasongt.com/projectpages/pipal.html"

        assert (root / "Train_Dist").exists(), \
            "Please place all distorted files into single folder named `Train_Dist`."

        # Read files with labels and merge them into single DF
        dfs = []
        for filename in (root / "Train_Label").glob("*.txt"):
            df = pd.read_csv(filename, index_col=None, header=None, names=['dist_img', 'score'])
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        df["ref_img"] = df["dist_img"].apply(lambda x: f"Train_Ref/{x[:5] + x[-4:]}")
        df["dist_img"] = df["dist_img"].apply(lambda x: f"Train_Dist/{x}")

        self.scores = df["score"].to_numpy()
        self.df = df[["dist_img", 'ref_img', 'score']]
        self.root = root


DATASETS = {
    "tid2013": TID2013,
    "kadid10k": KADID10k,
    "pipal": PIPAL,
}


def eval_metric(loader: DataLoader, metric: Metric, device: str, feature_extractor: str) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Evaluate metric on a given dataset.

    Args:
        loader: PyTorch dataloader that returns batch of distorted images, reference images and scores.
        metric: General metric that satisfies the Metric interface.
        device: Computation device.
        feature_extractor: name of the neural network to be used to extract features from images
    Returns:
        gt_scores: A tensor of ground truth values.
        metric_scores: A tensor of predicted values.
    """
    assert isinstance(loader, DataLoader), "Expect loader to be DataLoader class"
    assert isinstance(metric, Metric), f"Expected metric to be an instance of Metric, got {type(metric)} instead!"

    gt_scores = []
    metric_scores = []
    compute_function = determine_compute_function(metric_category=metric.category)

    for distorted_images, reference_images, scores in tqdm.tqdm(loader, ncols=50):
        distorted_images, reference_images = distorted_images.to(device), reference_images.to(device)
        gt_scores.append(scores.cpu())

        metric_score: torch.Tensor = \
            compute_function(metric.functor, distorted_images, reference_images, device, feature_extractor)

        if metric_score.dim() == 0:
            metric_score = metric_score.unsqueeze(0)

        metric_scores.append(metric_score.cpu())

    return torch.cat(gt_scores), torch.cat(metric_scores)


def determine_compute_function(metric_category: str) -> Callable:
    return {
        'FR': compute_full_reference,
        'NR': compute_no_reference,
        'DB': compute_distribution_based
    }[metric_category]


def get_feature_extractor(feature_extractor_name: str, device: str) -> nn.Module:
    r""" A factory to initialize feature extractor from its name. """
    if feature_extractor_name == "vgg16":
        return torchvision.models.vgg16(pretrained=True, progress=True).features.to(device)
    elif feature_extractor_name == "vgg19":
        return torchvision.models.vgg19(pretrained=True, progress=True).features.to(device)
    elif feature_extractor_name == "inception":
        return piq.feature_extractors.InceptionV3(
            resize_input=False, use_fid_inception=True, normalize_input=True).to(device)
    else:
        raise ValueError(f"Wrong feature extractor name {feature_extractor_name}")


def compute_full_reference(metric_functor: Callable, distorted_images: torch.Tensor,
                           reference_images: torch.Tensor, _, __) -> torch.Tensor:
    return metric_functor(distorted_images, reference_images).cpu()


def compute_no_reference(metric_functor: Callable, distorted_images: torch.Tensor, _, __, ___) -> torch.Tensor:
    return metric_functor(distorted_images).cpu()


def extract_features(distorted_patches: torch.Tensor, feature_extractor: nn.Module, feature_extractor_name: str,
                     reference_patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    distorted_features, reference_features = [], []
    with torch.no_grad():
        if feature_extractor_name == "inception":
            reference_features.append(feature_extractor(reference_patches)[0].squeeze())
            distorted_features.append(feature_extractor(distorted_patches)[0].squeeze())
        elif feature_extractor_name in ["vgg16", "vgg19"]:
            reference_features.append(torch.nn.functional.avg_pool2d(feature_extractor(reference_patches), 3).squeeze())
            distorted_features.append(torch.nn.functional.avg_pool2d(feature_extractor(distorted_patches), 3).squeeze())
        else:
            raise ValueError(f'Unknown feature extractor {feature_extractor_name} is selected. '
                             f'Please choose on of supported feature extractors: [inception, vgg16, vgg19]')

    distorted_features = torch.cat(distorted_features, dim=0)
    reference_features = torch.cat(reference_features, dim=0)

    return distorted_features, reference_features


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    r""" Map tensor values to [0, 1] """
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def compute_distribution_based(metric_functor: Callable, distorted_images: torch.Tensor,
                               reference_images: torch.Tensor, device: str, feature_extractor_name: str) \
        -> torch.Tensor:
    feature_extractor = get_feature_extractor(feature_extractor_name=feature_extractor_name, device=device)

    if feature_extractor_name == 'inception':
        distorted_images = normalize_tensor(distorted_images)
        reference_images = normalize_tensor(reference_images)

    # Create patches
    distorted_patches = crop_patches(distorted_images, size=96, stride=32)
    reference_patches = crop_patches(reference_images, size=96, stride=32)

    # Extract features from distorted images
    distorted_patches = distorted_patches.view(-1, *distorted_patches.shape[-3:])
    reference_patches = reference_patches.view(-1, *reference_patches.shape[-3:])

    distorted_features, reference_features = extract_features(distorted_patches, feature_extractor,
                                                              feature_extractor_name, reference_patches)

    return metric_functor(distorted_features, reference_features).cpu()


def crop_patches(images: torch.Tensor, size: int = 64, stride: int = 32) -> torch.Tensor:
    r"""Crop input images into smaller patches.

    Args:
        images: Tensor of images with shape (batch x 3 x H x W)
        size: size of a square patch
        stride: Step between patches
    Returns:
        A tensor on cropped patches of shape (-1, 3, size, size)
    """
    patches = images.data.unfold(1, 3, 3).unfold(2, size, stride).unfold(3, size, stride)
    patches = patches.reshape(-1, 3, size, size)
    return patches


def main(dataset_name: str, path: Path, metrics: List[str], batch_size: int, device: str, feature_extractor: str) \
        -> None:
    # Init dataset and dataloader
    dataset = DATASETS[dataset_name](root=path)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    # If category of metrics is selected instead of a list of metrics, take all metrics from this category
    if metrics[0] in METRIC_CATEGORIES:
        metrics = METRIC_CATEGORIES[metrics[0]]

    if metrics[0] == 'all':
        metrics = list(chain(*METRIC_CATEGORIES.values()))

    for metric_name in metrics:
        metric: Metric = METRICS[metric_name]
        gt_scores, metric_scores = eval_metric(loader, metric, device=device, feature_extractor=feature_extractor)
        gt_scores, metric_scores = gt_scores.numpy(), metric_scores.numpy()
        print(f"{metric_name}: SRCC {abs(spearmanr(gt_scores, metric_scores)[0]):0.3f}",
              f"KRCC {abs(kendalltau(gt_scores, metric_scores)[0]):0.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PIQ metrics")

    parser.add_argument("--dataset", type=str, help="Dataset name", choices=list(DATASETS.keys()))
    parser.add_argument("--path", type=Path, help="Path to dataset")
    parser.add_argument('--metrics', nargs='+', default=[], help='Metrics to benchmark',
                        choices=list(METRICS.keys()) + list(METRIC_CATEGORIES.keys()) + ['all'])
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Computation device')
    parser.add_argument('--feature_extractor', default='inception', choices=['inception', 'vgg16', 'vgg19'],
                        help='Select a feature extractor. For distribution-based metrics only')

    args = parser.parse_args()
    print(f"Parameters used for benchmark: {args}")

    main(
        dataset_name=args.dataset,
        path=args.path,
        metrics=args.metrics,
        batch_size=args.batch_size,
        device=args.device,
        feature_extractor=args.feature_extractor
    )
