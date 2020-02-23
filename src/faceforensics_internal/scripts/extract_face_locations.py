"""Extract all face bounding boxes from videos."""
import json
import multiprocessing as mp
from pathlib import Path

import click
import cv2
import face_recognition
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure


def extract_face_locations_from_video(video_path: Path, target_sub_dir: Path):

    output_path = (target_sub_dir / video_path.name).with_suffix(".json")
    if output_path.exists():
        return

    # Get video capture
    video_capture = cv2.VideoCapture(str(video_path))

    bounding_boxes = {}
    frame_count = -1
    while video_capture.isOpened():
        frame_count += 1
        # Read next frame
        ret, frame = video_capture.read()

        if not ret:
            video_capture.release()
            break

        face_locations = face_recognition.face_locations(frame)

        bounding_boxes[f"{frame_count:04d}"] = face_locations

    with open(output_path, "w") as f:
        json.dump(bounding_boxes, f)


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", "-c", multiple=True, default=[Compression.c40])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
@click.option("--cpu_count", required=False, type=click.INT, default=mp.cpu_count())
def extract_face_locations_from_videos(
    source_dir_root, compressions, methods, cpu_count
):
    # use FaceForensicsDataStructure to iterate over the correct image folders
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.videos,),
    )

    # this will be used to iterate the same way as the source dir
    # -> create same data structure again
    target_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.face_information,),
    )

    # zip source and target structure to iterate over both simultaneously
    for source_sub_dir, target_sub_dir in zip(
        source_dir_data_structure.get_subdirs(), target_dir_data_structure.get_subdirs()
    ):

        if not source_sub_dir.exists():
            continue

        target_sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {source_sub_dir.parts[-2]}, {source_sub_dir.parts[-3]}")

        # extract for each folder (-> video) the face information
        Parallel(n_jobs=cpu_count)(
            delayed(
                lambda _video_path: extract_face_locations_from_video(
                    _video_path, target_sub_dir
                )
            )(video_path)
            for video_path in tqdm(sorted(source_sub_dir.iterdir()))
        )


if __name__ == "__main__":
    extract_face_locations_from_videos()
