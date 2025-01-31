import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from .utils import DataProcessing, BEVBox3D

log = logging.getLogger(__name__)


class KITTI(BaseDataset):
    """This class is used to create a dataset based on the KITTI dataset, and
    used in object detection, visualizer, training, or testing.
    """

    def __init__(self,
                 dataset_path,
                 ground_point_range=[-100, 100],
                 downsample_step=1,
                 recenter_offset=[0,0],
                 name='KITTI',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            ground_point_range: list - The range of ground points to downsample
            downsample_step: int - The steps to skips per ground points
            recenter_offset: list - x and y offset value to shift point cloud range
            name: The name of the dataset (KITTI in this case).
            cache_dir: The directory where the cache is stored.
            test_result_folder: Path to store test output.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         ground_point_range=ground_point_range,
                         downsample_step=downsample_step,
                         recenter_offset=recenter_offset,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 2
        self.label_to_names = self.get_label_to_names()

        self.train_files = glob(
            join(cfg.dataset_path, 'training', 'velodyne', '*.bin'))
        self.train_files.sort()

        self.val_files = glob(
            join(cfg.dataset_path, 'validation', 'velodyne', '*.bin'))
        self.val_files.sort()        

        self.test_files = glob(
            join(cfg.dataset_path, 'testing', 'velodyne', '*.bin'))
        self.test_files.sort()

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """

        # change classes to suit the dataset needs
        label_to_names = {
            0: 'human',
            1: 'car',
            2: 'DontCare'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def read_label(path, calib, recenter_offset):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        if not Path(path).exists():
            return []

        with open(path, 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            label = line.strip().split(' ')

            center = np.array(
                [float(label[11]),
                 float(label[12]),
                 float(label[13]), 1.0])

            points = center @ np.linalg.inv(calib['world_cam'])

            # adjust x and y axis (velodyne) to centre

            points[0] = points[0] + recenter_offset[0]
            points[1] = points[1] + recenter_offset[1]

            size = [float(label[9]), float(label[8]), float(label[10])]  # w,h,l
            center = [points[0], points[1], size[1] / 2 + points[2]]

            objects.append(Object3d(center, size, label, calib))

        return objects

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate(
            [mat, np.array([[0., 0., 1., 0.]], dtype=mat.dtype)], axis=0)
        return mat

    @staticmethod
    def read_calib(path):
        """Reads calibiration for the dataset. You can use them to compare
        modeled results to observed results.

        Returns:
            The camera and the camera image used in calibration.
        """
        assert Path(path).exists()

        with open(path, 'r') as f:
            lines = f.readlines()

        obj = lines[0].strip().split(' ')[1:]
        P0 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[1].strip().split(' ')[1:]
        P1 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32).reshape(3, 4)

        P0 = KITTI._extend_matrix(P0)
        P1 = KITTI._extend_matrix(P1)
        P2 = KITTI._extend_matrix(P2)
        P3 = KITTI._extend_matrix(P3)

        obj = lines[4].strip().split(' ')[1:]
        rect_4x4 = np.eye(4, dtype=np.float32)
        rect_4x4[:3, :3] = np.array(obj, dtype=np.float32).reshape(3, 3)

        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.eye(4, dtype=np.float32)
        Tr_velo_to_cam[:3] = np.array(obj, dtype=np.float32).reshape(3, 4)

        world_cam = np.transpose(rect_4x4 @ Tr_velo_to_cam)
        cam_img = np.transpose(P2)

        return {'world_cam': world_cam, 'cam_img': cam_img}

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return KITTISplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        make_dir(self.cfg.test_result_folder)
        for attr, res in zip(attrs, results):
            name = attr['name']
            path = join(self.cfg.test_result_folder, name + '.txt')
            f = open(path, 'w')
            for box in res:
                f.write(box.to_kitti_format(box.confidence, self.cfg.recenter_offset))
                f.write('\n')


class KITTISplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Using custom dataset for Reliance")
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        label_path = pc_path.replace('velodyne',
                                     'label_2').replace('.bin', '.txt')
        calib_path = label_path.replace('label_2', 'calib')

        pc = self.dataset.read_lidar(pc_path)
        calib = self.dataset.read_calib(calib_path)
        label = self.dataset.read_label(label_path, calib, self.cfg.recenter_offset)

        # Downsample ground points
        reduced_pc = DataProcessing.downsample_ground_points(pc, self.cfg.ground_point_range, 
                        self.cfg.downsample_step)
                        
        # recenter point cloud
        reduced_pc = DataProcessing.recenter_pc(reduced_pc, self.cfg.recenter_offset)

        data = {
            'point': reduced_pc,
            'full_point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


class Object3d(BEVBox3D):
    """The class stores details that are object-specific, such as bounding box
    coordinates, occulusion and so on.
    """

    def __init__(self, center, size, label, calib=None):

        confidence = float(label[15]) if label.__len__() == 16 else -1.0

        world_cam = calib['world_cam']
        cam_img = calib['cam_img']

        # kitti boxes are pointing backwards
        yaw = float(label[14]) - np.pi
        yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi

        self.truncation = float(label[1])
        self.occlusion = float(
            label[2]
        )  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown

        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(
            label[6]), float(label[7])),
                              dtype=np.float32)

        class_name = label[0] if label[0] in KITTI.get_label_to_names().values(
        ) else 'DontCare'

        super().__init__(center, size, yaw, class_name, confidence, world_cam,
                         cam_img)

        self.yaw = float(label[14])

    def get_difficulty(self):
        """The method determines difficulty level of the object, such as Easy,
        Moderate, or Hard.
        """
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        # Add custom difficulity for data without first 7 values in kitti label files
        elif height >=1 and self.truncation == 0 and self.occlusion == 0:
            self.level_str = 'No Image'
            return 3
        else:
            self.level_str = 'UnKnown'
            return -1

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.label_class, self.truncation, self.occlusion, self.alpha, self.box2d, self.size[2], self.size[0], self.size[1],
                        self.center, self.yaw)
        return print_str


DATASET._register_module(KITTI)
