import json
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import click
import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure

logger = logging.getLogger(__file__)


def _calculate_tracking_bounding_box(
    face_bb: Dict[str, List[int]], image_size: Tuple[int, int], scale: int = 1.3
):
    height, width = image_size

    bounding_boxes = face_bb.values()
    x = min([bounding_box[0] for bounding_box in bounding_boxes])
    y = min([bounding_box[1] for bounding_box in bounding_boxes])
    w = max([bounding_box[2] for bounding_box in bounding_boxes])
    h = max([bounding_box[3] for bounding_box in bounding_boxes])

    size_bb = int(max(w, h) * scale)

    center_x, center_y = x + int(0.5 * w), y + int(0.5 * h)

    # Check for out of bounds, x-y lower left corner
    x = max(int(center_x - size_bb // 2), 0)
    y = max(int(center_y - size_bb // 2), 0)

    # Check for too big size for given x, y
    size_bb = min(width - x, size_bb)
    size_bb = min(height - y, size_bb)

    relative_bb = {}
    for key in face_bb.keys():
        _x, _y, w, h = face_bb[key]
        relative_bb[key] = _x - x, _y - y, w, h
        face_bb[key] = [x, y, size_bb, size_bb]

    return relative_bb


def _face_bb_to_tracked_bb(
    face_bb: Dict[str, List[int]], image_size: Tuple[int, int], scale: int = 1.3
):
    current_sequence = {}
    tracked_bb = {}
    relative_bb = {}

    def calculate_tracked_bb_for_sequence(image_name):
        if len(current_sequence) > 0:
            relative_bb.update(
                _calculate_tracking_bounding_box(
                    current_sequence, image_size, scale=scale
                )
            )
            tracked_bb.update(current_sequence)
        if image_name:
            relative_bb[image_name] = None
            tracked_bb[image_name] = None

    for image_name, face_bb_value in sorted(face_bb.items()):
        if not face_bb_value:
            calculate_tracked_bb_for_sequence(image_name)
            current_sequence = {}
        else:
            current_sequence[image_name] = face_bb_value

    if len(current_sequence) > 0:
        calculate_tracked_bb_for_sequence(None)
    return tracked_bb, relative_bb


def _extract_face(img_path, face, face_images_dir):
    if not face:
        return False
    img = cv2.imread(str(img_path))
    x, y, w, h = face
    cropped_face = img[y : y + h, x : x + w]  # noqa E203
    cv2.imwrite(str(face_images_dir / img_path.name), cropped_face)
    return True


def _get_image_size(video_folder: Path):
    height, width, _ = cv2.imread(str(next(video_folder.iterdir()))).shape
    return height, width


def _extract_faces_tracked_from_video(
    video_folder: Path, bounding_boxes: Path, face_images: Path
) -> bool:
    with open(str((bounding_boxes / video_folder.name).with_suffix(".json")), "r") as f:
        face_bb = json.load(f)

    face_images = face_images / video_folder.name
    face_images.mkdir(exist_ok=True)

    tracked_bb, relative_bb = _face_bb_to_tracked_bb(
        face_bb, image_size=_get_image_size(video_folder)
    )

    # extract all faces and save it
    for img in sorted(video_folder.iterdir()):
        face = tracked_bb[img.with_suffix("").name]
        _extract_face(img, face, face_images)

    # save relative face positions as well
    with open(face_images / "relative_bb.json", "w") as f:
        json.dump(relative_bb, f)

    return True


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", "-c", multiple=True, default=[Compression.c40])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
def extract_faces_tracked(source_dir_root, compressions, methods):
    full_images_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.full_images,),
        methods=methods,
    )

    bounding_boxes_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.bounding_boxes,),
        methods=methods,
    )

    face_images_tracked_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.face_images_tracked,),
        methods=methods,
    )

    for full_images, bounding_boxes, face_images in zip(
        full_images_data_structure.get_subdirs(),
        bounding_boxes_dir_data_structure.get_subdirs(),
        face_images_tracked_dir_data_structure.get_subdirs(),
    ):
        logger.info(f"Current method: {full_images.parents[1].name}")

        face_images.mkdir(exist_ok=True)

        # extract faces from videos in parallel
        Parallel(n_jobs=12)(
            delayed(
                lambda _video_folder: _extract_faces_tracked_from_video(
                    _video_folder, bounding_boxes, face_images
                )
            )(video_folder)
            for video_folder in tqdm(sorted(full_images.iterdir()))
        )


if __name__ == "__main__":
    extract_faces_tracked()
