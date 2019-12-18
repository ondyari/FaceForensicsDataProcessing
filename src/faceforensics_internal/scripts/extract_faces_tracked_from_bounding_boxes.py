import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import click
import cv2
from cv2.cv2 import VideoCapture
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
    left = min([bounding_box[0] for bounding_box in bounding_boxes])
    top = min([bounding_box[1] for bounding_box in bounding_boxes])
    right = max([bounding_box[0] + bounding_box[2] for bounding_box in bounding_boxes])
    bottom = max([bounding_box[1] + bounding_box[3] for bounding_box in bounding_boxes])

    x, y, w, h = left, top, right - left, bottom - top

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


def _extract_face(img, face, face_images_dir, frame_number):
    if not face:
        return False
    x, y, w, h = face
    try:
        cropped_face = img[y : y + int(h), x : x + int(w)]  # noqa E203
    except TypeError:
        print(face)
        raise
    cv2.imwrite(str(face_images_dir / f"{frame_number:04d}.png"), cropped_face)
    return True


def _get_image_size(video_folder: Path):
    height, width, _ = cv2.imread(str(next(video_folder.iterdir()))).shape
    return height, width


def _extract_faces_tracked_from_video(
    video_folder: Path, bounding_boxes: Path, face_images: Path
) -> bool:
    with open(str((bounding_boxes / video_folder.name).with_suffix(".json")), "r") as f:
        face_bb = json.load(f)

    face_images = face_images / video_folder.with_suffix("").name
    face_images.mkdir(exist_ok=True)

    cap = VideoCapture(str(video_folder))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracked_bb, relative_bb = _face_bb_to_tracked_bb(
        face_bb, image_size=(height, width), scale=1
    )

    # extract all faces and save it
    frame_num = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        face = tracked_bb[f"{frame_num:04d}"]
        _extract_face(image, face, face_images, frame_num)

        frame_num += 1
    cap.release()

    # save relative face positions as well
    with open(face_images / "relative_bb.json", "w") as f:
        json.dump(relative_bb, f)

    with open(face_images / "tracked_bb.json", "w") as f:
        json.dump(tracked_bb, f)

    return True


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", "-c", multiple=True, default=[Compression.c40])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
def extract_faces_tracked(source_dir_root, compressions, methods):
    videos_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.videos,),
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

    for videos, bounding_boxes, face_images in zip(
        videos_data_structure.get_subdirs(),
        bounding_boxes_dir_data_structure.get_subdirs(),
        face_images_tracked_dir_data_structure.get_subdirs(),
    ):
        logger.info(f"Current method: {videos.parents[1].name}")

        face_images.mkdir(exist_ok=True)

        # extract faces from videos in parallel
        Parallel(n_jobs=mp.cpu_count())(
            delayed(
                lambda _video_folder: _extract_faces_tracked_from_video(
                    _video_folder, bounding_boxes, face_images
                )
            )(video_folder)
            for video_folder in tqdm(sorted(videos.iterdir()))
        )


if __name__ == "__main__":
    extract_faces_tracked()
