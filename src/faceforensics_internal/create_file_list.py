from pathlib import Path
from typing import List

import click
import numpy as np

from faceforensics_internal.file_list_dataset import FileList
from faceforensics_internal.splits import TEST
from faceforensics_internal.splits import TEST_NAME
from faceforensics_internal.splits import TRAIN
from faceforensics_internal.splits import TRAIN_NAME
from faceforensics_internal.splits import VAL
from faceforensics_internal.splits import VAL_NAME
from faceforensics_internal.utils import _img_name_to_int
from faceforensics_internal.utils import cl_logger
from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure


def _get_min_sequence_length(source_dir_data_structure):
    min_length = -1

    for source_sub_dir in source_dir_data_structure.get_subdirs():
        for video_folder in sorted(source_sub_dir.iterdir()):
            number_of_frames = len(list(video_folder.glob("*.png")))
            if min_length == -1 or min_length > number_of_frames:
                min_length = number_of_frames

    return min_length


def _select_frames(nb_images: int, samples_per_video: int) -> List[int]:
    """Selects frames to take from video.

    Args:
        nb_images: length of video aka. number of frames in video
        samples_per_video: how many frames of this video should be taken. If this value
            is bigger then nb_images or -1, nb_images are taken.

    """
    if samples_per_video == -1 or samples_per_video > nb_images:
        selected_frames = range(nb_images)
    else:
        selected_frames = np.rint(
            np.linspace(1, nb_images, min(samples_per_video, nb_images)) - 1
        ).astype(int)
    return selected_frames


def _create_file_list(
    methods,
    compressions,
    data_types,
    min_sequence_length,
    output_file,
    samples_per_video,
    source_dir_root,
):
    file_list = FileList(
        root=source_dir_root, classes=methods, min_sequence_length=min_sequence_length
    )
    # use faceforensicsdatastructure to iterate elegantly over the correct
    # image folders
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=compressions,
        data_types=data_types,
    )

    _min_sequence_length = _get_min_sequence_length(source_dir_data_structure)
    if _min_sequence_length < samples_per_video:
        cl_logger.warning(
            f"There is a sequence that is sequence that has less frames "
            f"then you would like to sample: "
            f"{_min_sequence_length}<{samples_per_video}"
        )

    for split, split_name in [(TRAIN, TRAIN_NAME), (VAL, VAL_NAME), (TEST, TEST_NAME)]:
        for source_sub_dir, target in zip(
            source_dir_data_structure.get_subdirs(), file_list.classes
        ):
            for video_folder in sorted(source_sub_dir.iterdir()):
                if video_folder.name.split("_")[0] in split:

                    images = sorted(video_folder.glob("*.png"))
                    filtered_images_idx = []

                    # find all frames that have at least min_sequence_length-1 preceeding
                    # frames
                    sequence_start = _img_name_to_int(images[0])
                    last_idx = sequence_start
                    for list_idx, image in enumerate(images):
                        image_idx = _img_name_to_int(image)
                        if last_idx + 1 != image_idx:
                            sequence_start = image_idx
                        elif image_idx - sequence_start >= min_sequence_length - 1:
                            filtered_images_idx.append(list_idx)
                        last_idx = image_idx

                    # for the test-set all frames are going to be taken
                    # otherwise distribute uniformly
                    selected_frames = _select_frames(
                        len(filtered_images_idx),
                        -1 if split_name == TEST_NAME else samples_per_video,
                    )

                    sampled_images_idx = np.asarray(filtered_images_idx)[
                        selected_frames
                    ]
                    file_list.add_data_points(
                        path_list=images,
                        target_label=target,
                        split=split_name,
                        sampled_images_idx=sampled_images_idx,
                    )

    file_list.save(output_file)
    cl_logger.info(f"{output_file} created.")
    return file_list


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option(
    "--target_dir_root",
    default=None,
    help="If specified, all files in the filelist are copied over to this location",
)
@click.option("--output_file", required=True, type=click.Path())
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
@click.option("--compressions", "-c", multiple=True, default=[Compression.c40])
@click.option(
    "--data_types", "-d", multiple=True, default=[DataType.face_images_tracked]
)
@click.option(
    "--samples_per_video",
    "-s",
    default=-1,
    help="Number of frames selected per video. For videos with less frames then this"
    "number, only these are selected. If samples_per_video is -1 all frames for each"
    "video is selected.",
)
@click.option(
    "--min_sequence_length",
    default=1,
    help="Indicates how many preceeded consecutive frames make a frame eligible (i.e."
    "if set to 5 frame 0004 is eligible if frames 0000-0003 are present as well.",
)
def create_file_list(
    source_dir_root,
    target_dir_root,
    output_file,
    methods,
    compressions,
    data_types,
    samples_per_video,
    min_sequence_length,
):

    try:
        # if file exists, we don't have to create it again
        file_list = FileList.load(output_file)
    except FileNotFoundError:
        file_list = _create_file_list(
            methods,
            compressions,
            data_types,
            min_sequence_length,
            output_file,
            samples_per_video,
            source_dir_root,
        )

    if target_dir_root:
        file_list.copy_to(Path(target_dir_root))
        file_list.save(output_file)

    for split in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
        data_set = FileList.get_dataset_form_file(output_file, split)
        cl_logger.info(f"{split}-data-set: {data_set}")


if __name__ == "__main__":
    create_file_list()