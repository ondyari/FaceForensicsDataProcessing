import itertools
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union


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


class Method:
    REAL_DIR = "original_sequences/"
    FAKE_DIR = "manipulated_sequences/"

    def __init__(self, name: str, is_real: bool):
        self.name = name
        self.is_real = is_real

    def get_dir_str(self):
        if self.is_real:
            return self.REAL_DIR + self.name
        else:
            return self.FAKE_DIR + self.name

    def __str__(self):
        return self.name


ACTORS = Method("actors", is_real=True)
YOUTUBE = Method("youtube", is_real=True)

DEEP_FAKE_DETECTION = Method("DeepFakeDetection", is_real=False)
DEEPFAKES = Method("Deepfakes", is_real=False)
FACE2FACE = Method("Face2Face", is_real=False)
FACE_SWAP = Method("FaceSwap", is_real=False)
NEURAL_TEXTURES = Method("NeuralTextures", is_real=False)


class FaceForensicsDataStructure:

    METHODS = {
        ACTORS.name: ACTORS,
        YOUTUBE.name: YOUTUBE,
        DEEP_FAKE_DETECTION.name: DEEP_FAKE_DETECTION,
        DEEPFAKES.name: DEEPFAKES,
        FACE2FACE.name: FACE2FACE,
        FACE_SWAP.name: FACE_SWAP,
        NEURAL_TEXTURES.name: NEURAL_TEXTURES,
    }

    ALL_METHODS = list(METHODS.keys())

    def __init__(
        self,
        root_dir: str,
        methods: Tuple[str, ...],
        compressions: Iterable[Union[str, Compression]] = (Compression.raw,),
        data_types: List[Union[str, DataType]] = (DataType.face_images,),
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist!")
        self.methods = [self.METHODS[method] for method in methods]
        self.data_types = data_types
        self.compressions = compressions

    def get_subdirs(self) -> List[Path]:

        return [
            self.root_dir / method.get_dir_str() / str(compression) / str(data_type)
            for method, compression, data_type in itertools.product(
                self.methods, self.compressions, self.data_types
            )
        ]
