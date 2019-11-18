"""Extract all face bounding boxes from videos."""
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


def extract_bounding_boxes_from_video(
    video_path, target_sub_dir, face_detector_model, max_num_frames
):

    output_path = Path(target_sub_dir / video_path.name).with_suffix(".json")
    if output_path.exists():
        return

    # Get video capture
    video_capture = cv2.VideoCapture(str(video_path))
    max_num_frames = min(
        max_num_frames, int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    )

    bounding_boxes = {}
    frame_count = -1
    while video_capture.isOpened():
        frame_count += 1

        # Read next frame
        ret, frame = video_capture.read()

        if not ret:
            video_capture.release()
            break

        if frame_count == max_num_frames:
            video_capture.release()
            break

        face_locations = face_recognition.face_locations(
            frame, model=face_detector_model
        )

        bounding_boxes[f"{frame_count:04d}"] = face_locations

    print(bounding_boxes)
    # with open(output_path, "w") as f:
    #     json.dump(bounding_boxes, f)


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--target_dir_root", required=True, type=click.Path(exists=False))
def extract_bounding_boxes_from_videos(
    source_dir_root, target_dir_root, face_detector_model="hog"
):
    target_dir_root = Path(target_dir_root)
    target_dir_root.mkdir(parents=True, exist_ok=True)

    for compression in Compression:

        # use FaceForensicsDataStructure to iterate elegantly over the correct image folders
        source_dir_data_structure = FaceForensicsDataStructure(
            source_dir_root,
            compression=str(compression),
            data_type=str(DataType.videos),
        )

        # this will be used to iterate the same way as the source dir -> create same data structure again
        target_dir_data_structure = FaceForensicsDataStructure(
            target_dir_root,
            compression=str(compression),
            data_type=str(DataType.bounding_boxes),
        )

        # zip all 3 together and iterate
        for source_sub_dir, target_sub_dir in zip(
            source_dir_data_structure.get_subdirs(),
            target_dir_data_structure.get_subdirs(),
        ):

            if not source_sub_dir.exists():
                continue

            target_sub_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing {source_sub_dir.parts[-2]}, {source_sub_dir.parts[-3]}")

            # extract for each folder (-> video) the face information
            Parallel(n_jobs=8)(
                delayed(
                    lambda _video_path: extract_bounding_boxes_from_video(
                        _video_path,
                        target_sub_dir,
                        face_detector_model,
                        max_num_frames=3,
                    )
                )(video_path)
                for video_path in tqdm(sorted(source_sub_dir.iterdir())[:3])
            )


if __name__ == "__main__":
    extract_bounding_boxes_from_videos()
