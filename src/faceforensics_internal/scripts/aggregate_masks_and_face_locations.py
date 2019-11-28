import json
import logging
import math
from json import JSONDecodeError
from pathlib import Path
from typing import Union

import click
from joblib import delayed
from joblib import Parallel
from joblib._multiprocessing_helpers import mp
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure

logger = logging.getLogger(__file__)


def _largest_face_location(face_location):
    x, y, w, h = face_location
    bb_size = max(w, h)
    return bb_size


def closest_center(face_location_0, face_location_1):
    distance = get_distance(
        face_location_to_center(face_location_1),
        face_location_to_center(face_location_0),
    )
    return distance


def close_enough(face_location_0, face_location_1, min_distance=1.0 / 5):
    x, y, w, h = face_location_0
    distance = get_distance(
        face_location_to_center(face_location_1),
        face_location_to_center(face_location_0),
    )
    bb_size = max(w, h)

    return distance < bb_size * min_distance


def get_distance(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    distance = math.hypot(x2 - x1, y2 - y1)

    return distance


def get_iou(bb1, bb2):
    bb1 = {"x1": bb1[0], "x2": bb1[0] + bb1[2], "y1": bb1[1], "y2": bb1[1] + bb1[3]}

    bb2 = {"x1": bb2[0], "x2": bb2[0] + bb2[2], "y1": bb2[1], "y2": bb2[1] + bb2[3]}
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def face_location_to_center(face_location):
    """Compute center of face_location given in (x, y, w, h) format."""

    x, y, w, h = face_location
    return x + int(0.5 * w), y + int(0.5 * h)


def trbl_to_xywh(face_location_trbl):

    top, right, bottom, left = face_location_trbl

    x = left
    y = top
    w = right - x
    h = bottom - y

    return [x, y, w, h]


def _filter_face_information(face_information: Path, masks: Union[Path, None], output):
    last_location = None
    resulting_face_locations = {}

    with open(str(face_information), "r") as f:
        try:
            face_information_json = json.load(f)
        except JSONDecodeError:
            logger.error(f"Could not read json: {face_information}")
            return

    if masks:
        if face_information.suffix != masks.suffix:
            logger.error(
                f"face_information path and mask path are not equal!\n"
                f"{face_information} != {masks},"
            )
            return

        with open(str(masks), "r") as f:
            masks_json = json.load(f)

        for frame, locations in face_information_json.items():
            # some file contain additional information besides the face_locations
            try:
                if len(locations[1]) == 0:
                    locations = locations[0]
                elif isinstance(locations[1][0], dict):
                    locations = locations[0]
            except IndexError:
                pass

            # if there is no face set last_location to None
            if len(locations) == 0:
                last_location = None
                resulting_face_locations[frame] = last_location
                continue

            locations = map(trbl_to_xywh, locations)

            if not last_location:
                locations = sorted(locations, reverse=True, key=_largest_face_location)
                mask_bounding_box = masks_json[frame]
                if len(mask_bounding_box) == 0:
                    last_location = locations[0]
                    resulting_face_locations[frame] = last_location
                    continue
                else:
                    mask_bounding_box = trbl_to_xywh(mask_bounding_box[0])

                iou = get_iou(mask_bounding_box, locations[0])

                if iou > 0.1:
                    last_location = locations[0]  # largest face
                    resulting_face_locations[frame] = last_location
                    continue
                else:
                    # in all other cases add the mask
                    last_location = mask_bounding_box
                    resulting_face_locations[frame] = mask_bounding_box
                    continue

            locations = sorted(
                locations, key=lambda location: closest_center(last_location, location)
            )

            mask_bounding_box = masks_json[frame]
            # if there is no mask make sure the largest face is close enough to the last
            # face tracked
            if len(mask_bounding_box) == 0:
                if close_enough(
                    last_location, locations[0]
                ):  # also if the face is close
                    # enough add it
                    last_location = locations[0]  # largest face
                    resulting_face_locations[frame] = last_location
                    continue
                else:
                    last_location = None
                    resulting_face_locations[frame] = None
                    continue
            else:
                mask_bounding_box = trbl_to_xywh(mask_bounding_box[0])

            iou = get_iou(mask_bounding_box, locations[0])

            # if iou is big enough its possible to add face
            if iou > 0.1:
                if close_enough(
                    last_location, locations[0]
                ):  # also if the face is close
                    # enough add it
                    last_location = locations[0]  # largest face
                    resulting_face_locations[frame] = last_location
                    continue
                else:
                    last_location = None
                    resulting_face_locations[frame] = None
                    continue

            # in all other cases add the mask
            last_location = mask_bounding_box
            resulting_face_locations[frame] = mask_bounding_box

    # without masks
    else:
        for frame, locations in face_information_json.items():
            # some file contain additional information besides the face_locations
            try:
                if len(locations[1]) == 0:
                    locations = locations[0]
                elif isinstance(locations[1][0], dict):
                    locations = locations[0]
            except IndexError:
                pass

            # if there is no face set last_location to None
            if len(locations) == 0:
                last_location = None
                resulting_face_locations[frame] = last_location
                continue

            locations = map(trbl_to_xywh, locations)

            if not last_location:
                locations = sorted(locations, reverse=True, key=_largest_face_location)
                last_location = locations[0]  # largest face
                resulting_face_locations[frame] = last_location
                continue

            locations = sorted(
                locations, key=lambda location: closest_center(last_location, location)
            )

            # if its close enough add it
            if close_enough(last_location, locations[0]):
                last_location = locations[0]
                resulting_face_locations[frame] = last_location
            else:
                last_location = None
                resulting_face_locations[frame] = last_location

    with open(str(output / face_information.name), "w") as f:
        json.dump(resulting_face_locations, f)


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
@click.option("--compression", "-c", default=Compression.c40)
def aggregate_masks_and_face_locations(source_dir_root, methods, compression):

    face_information_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=(compression,),
        data_types=(DataType.face_information,),
    )

    bounding_boxs_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=(compression,),
        data_types=(DataType.bounding_boxes,),
    )

    mask_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=(Compression.masks,),
        data_types=(DataType.bounding_boxes,),
    )

    for face_information, bounding_boxes, mask_data in zip(
        face_information_data_structure.get_subdirs(),
        bounding_boxs_data_structure.get_subdirs(),
        mask_data_structure.get_subdirs(),
    ):

        if not face_information.exists():
            continue

        bounding_boxes.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Processing {face_information.parts[-2]}, {face_information.parts[-3]}"
        )

        if not mask_data.exists():
            logging.info("Didn't find any mask data.")

            # compute mask bounding box for each folder
            Parallel(n_jobs=mp.cpu_count())(
                delayed(
                    lambda _face_information_video: _filter_face_information(
                        _face_information_video, None, bounding_boxes
                    )
                )(face_information_video)
                for face_information_video in tqdm(sorted(face_information.iterdir()))
            )

        else:
            # compute mask bounding box for each folder
            Parallel(n_jobs=mp.cpu_count())(
                delayed(
                    lambda _face_information_video, _mask_data_video: _filter_face_information(
                        _face_information_video, _mask_data_video, bounding_boxes
                    )
                )(face_information_video, mask_data_video)
                for face_information_video, mask_data_video in tqdm(
                    zip(sorted(face_information.iterdir()), sorted(mask_data.iterdir()))
                )
            )


if __name__ == "__main__":
    aggregate_masks_and_face_locations()
