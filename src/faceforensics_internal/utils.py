from enum import auto, Enum
from pathlib import Path
from typing import List


class StrEnum(Enum):
    def __str__(self):
        return self.name


class Compression(StrEnum):
    raw = auto()
    c23 = auto()
    c40 = auto()


class DataType(StrEnum):
    bounding_boxes = auto()
    face_images = auto()
    face_images_tracked = auto()
    full_images = auto()
    face_information = auto()
    masks = auto()
    videos = auto()


class FaceForensicsDataStructure:
    SUB_DIRECTORIES = ["original_sequences/youtube"] + [
        "manipulated_sequences/" + manipulated_sequence
        for manipulated_sequence in [
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
        ]
    ]

    def __init__(
        self, root_dir: str, compression: str = "raw", data_type: str = "images"
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist!")
        self.data_type = data_type
        self.compression = compression

    def get_subdirs(self) -> List[Path]:
        return [
            self.root_dir / subdir / self.compression / self.data_type
            for subdir in self.SUB_DIRECTORIES
        ]
