import copy
import json
import logging
from pathlib import Path
from pprint import pformat
from typing import List

import numpy as np
import torch
from flowiz import read_flow
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from faceforensics_internal.splits import TEST_NAME
from faceforensics_internal.splits import TRAIN_NAME
from faceforensics_internal.splits import VAL_NAME

logger = logging.getLogger(__file__)


class FileList:
    def __init__(
        self,
        root: str,
        classes: List[str],
        min_sequence_length: int,
        flow: bool,
        image_size: int,
    ):
        self.root = root
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples_face_images_paths = {TRAIN_NAME: [], VAL_NAME: [], TEST_NAME: []}
        self.samples_idx = {TRAIN_NAME: [], VAL_NAME: [], TEST_NAME: []}

        self.min_sequence_length = min_sequence_length
        self.flow = flow
        self.image_size = image_size

    def binarize(self):
        authentic = self.classes[0]
        fake = "_".join(self.classes[1:])
        self.classes = [authentic, fake]
        self.class_to_idx = {authentic: 0, fake: 1}

        for data_points in self.samples_face_images_paths.values():
            for datapoint in data_points:
                binary_label = 0 if datapoint[1] == 0 else 1
                datapoint[1] = binary_label

    def add_face_image_data_point(
        self, path_face_image: Path, target_label: str, split: str
    ):
        self.samples_face_images_paths[split].append(
            (
                str(path_face_image.relative_to(self.root)),
                self.class_to_idx[target_label],
            )
        )

    def add_data_points(
        self,
        paths_face_images: List[Path],
        target_label: str,
        split: str,
        sampled_images_idx: np.array,
    ):
        nb_samples_offset = len(self.samples_face_images_paths[split])
        sampled_images_idx = (sampled_images_idx + nb_samples_offset).tolist()
        self.samples_idx[split] += sampled_images_idx

        for path_face_image in paths_face_images:
            self.add_face_image_data_point(path_face_image, target_label, split)

    def save(self, path):
        """Save self.__dict__ as json."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f)  # be careful with self.root->Path

    @classmethod
    def load(cls, path):
        """Restore instance from json via self.__dict__."""
        with open(path, "r") as f:
            __dict__ = json.load(f)
        file_list = cls.__new__(cls)
        file_list.__dict__.update(__dict__)
        return file_list

    def get_dataset(self, split, transform=None, sequence_length: int = 1) -> Dataset:
        """Get dataset by using this instance."""
        if sequence_length > self.min_sequence_length:
            logger.warning(
                f"{sequence_length}>{self.min_sequence_length}. Trying to load data that"
                f"does not exist might raise an error in the FileListDataset."
            )
        transform = transform or []
        transform = transforms.Compose(transform)
        return FileListDataset(
            file_list=self,
            split=split,
            sequence_length=sequence_length,
            transform=transform,
        )

    @classmethod
    def get_dataset_form_file(
        cls, path, split, transform=None, sequence_length: int = 1
    ) -> Dataset:
        """Get dataset by loading a FileList and calling get_dataset on it."""
        return cls.load(path).get_dataset(split, transform, sequence_length)

    def __str__(self):
        return pformat(self.class_to_idx, indent=4)


class FileListDataset(VisionDataset):
    """Almost the same as DatasetFolder by pyTorch.

    But this one does not build up a file list by walking a folder. Instead this file
    list has to be provided."""

    def __init__(
        self,
        file_list: FileList,
        split: str,
        sequence_length: int,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            file_list.root, transform=transform, target_transform=target_transform
        )
        self.classes = file_list.classes
        self.class_to_idx = file_list.class_to_idx
        self._samples_face_images_paths = file_list.samples_face_images_paths[split]
        self.samples_idx = file_list.samples_idx[split]
        self.targets = [s[1] for s in self._samples_face_images_paths]
        self.sequence_length = sequence_length
        self.flow = file_list.flow
        self.image_size = file_list.image_size

        self.loader_image = default_loader

        if self.flow:
            self.loader_flow = read_flow

    def __getitem__(self, index):
        index = self.samples_idx[index]

        # Get preceding face images paths
        samples_face_images_paths = self._samples_face_images_paths[
            index - self.sequence_length + 1 : index + 1  # noqa: 203
        ]

        if self.flow:
            # Get corresponding flow files paths
            samples_flow_files_paths = copy.deepcopy(samples_face_images_paths)
            for sample_flow_file_path in samples_flow_files_paths:
                sample_flow_file_path[0] = Path(
                    sample_flow_file_path[0].replace(
                        f"face_images_tracked_{self.image_size}",
                        f"flow_files_{self.image_size}",
                    )
                ).with_suffix(".flo")

        sample_face_image_first_path = samples_face_images_paths[0][0]
        target = samples_face_images_paths[0][1]

        # Load face images
        samples_face_images = []
        for sample_face_image_path in samples_face_images_paths:
            try:
                sample_face_image = self.loader_image(
                    f"{self.root}/{sample_face_image_path[0]}"
                )
            except OSError:
                logger.warning(f"{OSError}: {sample_face_image_path[0]}")
                sample_face_image = self.loader_image(
                    f"{self.root}/{sample_face_image_first_path}"
                )
            samples_face_images.append(sample_face_image)

        # Load flow files
        if self.flow:
            samples_flow_files = []
            for sample_flow_file_path in samples_flow_files_paths:
                try:
                    sample_flow_file = self.loader_flow(
                        f"{self.root}/{sample_flow_file_path[0]}"
                    )
                except OSError:
                    logger.warning(f"{OSError}: {sample_flow_file_path[0]}")
                    sample_flow_file = self.loader_image(
                        f"{self.root}/{samples_flow_files_paths[0][0]}"
                    )
                samples_flow_files.append(sample_flow_file)

        if self.sequence_length == 1:
            samples_face_images = samples_face_images[0]

            if self.flow:
                samples_flow_files = samples_flow_files[0]

        # Transform face images
        if self.transform is not None:
            if self.flow and self.sequence_length != 1:
                samples_face_images = list(map(self.transform, samples_face_images))
                samples_flow_files = list(map(ToTensor(), samples_flow_files))
            else:
                samples_face_images = self.transform(samples_face_images)

        if self.flow:
            # Concatenate face images and flow files along channels dimension
            samples_concatenated = []
            for sample_face_image, sample_flow_file in zip(
                samples_face_images, samples_flow_files
            ):
                sample_concatenated = torch.cat([sample_face_image, sample_flow_file])
                samples_concatenated.append(sample_concatenated)

        if self.sequence_length == 1:
            if self.flow:
                samples = samples_concatenated[0]
            else:
                samples = samples_face_images
        else:
            if self.flow:
                samples = torch.stack(samples_concatenated, dim=1)
            else:
                samples = torch.stack(samples_face_images, dim=1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        path_parts = Path(sample_face_image_first_path).parts
        video_name = "_".join([path_parts[1], path_parts[2], path_parts[4]])

        return samples, target, video_name

    def __len__(self):
        return len(self.samples_idx)
