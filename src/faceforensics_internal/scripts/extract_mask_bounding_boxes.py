"""Extract mask bounding box from mask video indicating the manipulated area."""
import json
import multiprocessing as mp
from pathlib import Path

import click
import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from faceforensics_internal.utils import get_mask_bounding_box
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure


def extract_bounding_boxes_from_video(video_path: Path, target_sub_dir: Path):

    output_path = (target_sub_dir / video_path.name).with_suffix(".json")
    #if output_path.exists():
    #    return

    # Get video capture
    mask_capture = cv2.VideoCapture(str(video_path))

    bounding_boxes = {}
    frame_count = -1
    while mask_capture.isOpened():
        frame_count += 1

        # Read next frame
        ret, mask = mask_capture.read()

        if not ret:
            mask_capture.release()
            break

        mask_bounding_box = get_mask_bounding_box(mask)

        if mask_bounding_box is not None:
            bounding_boxes[f"{frame_count:04d}"] = [list(mask_bounding_box)]
        else:
            print(video_path, frame_count, "no bounding box")
            return

    with open(output_path, "w") as f:
        json.dump(bounding_boxes, f)


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--target_dir_root", required=True, type=click.Path(exists=False))
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
def extract_bounding_box_from_masks(
    source_dir_root, target_dir_root, methods
):
    target_dir_root = Path(target_dir_root)
    target_dir_root.mkdir(parents=True, exist_ok=True)

    # use FaceForensicsDataStructure to iterate over the correct image folders
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=(DataType.masks,),
        data_types=(DataType.videos,),
    )

    # this will be used to iterate the same way as the source dir
    # -> create same data structure again
    target_dir_data_structure = FaceForensicsDataStructure(
        target_dir_root,
        methods=methods,
        compressions=(DataType.masks,),
        data_types=(DataType.bounding_boxes,),
    )

    # zip source and target structure to iterate over both simultaneously
    for source_sub_dir, target_sub_dir in zip(
        source_dir_data_structure.get_subdirs(), target_dir_data_structure.get_subdirs()
    ):

        if not source_sub_dir.exists():
            continue

        target_sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {source_sub_dir.parts[-2]}, {source_sub_dir.parts[-3]}")

        # compute mask bounding box for each folder
        Parallel(n_jobs=mp.cpu_count())(
            delayed(
                lambda _video_path: extract_bounding_boxes_from_video(
                    _video_path, target_sub_dir
                )
            )(video_path)
            for video_path in tqdm(sorted(source_sub_dir.iterdir()))
        )


if __name__ == "__main__":
    extract_bounding_box_from_masks()
