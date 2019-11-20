from enum import auto
from enum import Enum
from pathlib import Path
from typing import List
from typing import Tuple


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
    def __init__(
        self,
        root_dir: str,
        methods: Tuple[str, ...],
        compression: str = "raw",
        data_type: str = "images",
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist!")
        self.method_dirs = self.get_method_dirs(methods)
        self.data_type = data_type
        self.compression = compression

    def get_method_dirs(self, methods) -> List[str]:
        method_dirs = []

        for method in methods:
            if method == "youtube":
                sequence = "original_sequences/"
            else:
                sequence = "manipulated_sequences/"

            method_dirs.append(sequence + method)

        return sorted(method_dirs)

    def get_subdirs(self) -> List[Path]:

        return [
            self.root_dir / method_dir / self.compression / self.data_type
            for method_dir in self.method_dirs
        ]
