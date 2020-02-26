import logging
from pathlib import Path
from typing import List

import click
import numpy as np

from faceforensics_internal.file_list_dataset import FileList
from faceforensics_internal.splits import TEST
from faceforensics_internal.splits import TEST_AIF
from faceforensics_internal.splits import TEST_NAME
from faceforensics_internal.splits import TEST_NAME_AIF
from faceforensics_internal.splits import TRAIN
from faceforensics_internal.splits import TRAIN_NAME
from faceforensics_internal.splits import VAL
from faceforensics_internal.splits import VAL_NAME
from faceforensics_internal.utils import _img_name_to_int
from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure

logger = logging.getLogger(__file__)


def _get_min_sequence_length(source_dir_data_structure):
    min_length = -1

    for source_sub_dir in source_dir_data_structure.get_subdirs():
        for video_folder in sorted(source_sub_dir.iterdir()):
            number_of_frames = len(list(video_folder.glob("*.png")))
            if min_length == -1 or min_length > number_of_frames:
                logger.warning(
                    f"{video_folder.name} has only {number_of_frames} frames!"
                )
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
    source_dir_root,
    output_file,
    methods,
    compressions,
    data_types,
    samples_per_video_train,
    samples_per_video_val,
    samples_per_video_test,
    min_sequence_length,
    aif,
    flow,
    image_size,
):
    file_list = FileList(
        root=source_dir_root,
        classes=methods,
        min_sequence_length=min_sequence_length,
        flow=flow,
        image_size=image_size,
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
    if (
        _min_sequence_length < samples_per_video_train
        or _min_sequence_length < samples_per_video_val
        or _min_sequence_length < samples_per_video_test
    ):
        logger.warning(
            f"There is a sequence that has less frames "
            f"then you would like to sample: {_min_sequence_length}"
        )

    if aif:
        splits = [(TRAIN, TRAIN_NAME), (VAL, VAL_NAME), (TEST_AIF, TEST_NAME_AIF)]
    else:
        splits = [(TRAIN, TRAIN_NAME), (VAL, VAL_NAME), (TEST, TEST_NAME)]

    for split, split_name in splits:
        for source_sub_dir in source_dir_data_structure.get_subdirs():
            target = source_sub_dir.parts[-3]
            for video_folder in sorted(source_sub_dir.iterdir()):
                video_name = video_folder.name

                if aif:
                    split_check = video_name
                else:
                    split_check = video_name.split("_")[0]

                if split_check in split:
                    paths_face_images = sorted(video_folder.glob("*.png"))
                    filtered_images_idx = []

                    # find all frames that have at least min_sequence_length-1 preceeding
                    # frames
                    if len(paths_face_images) == 0:
                        continue

                    sequence_start = _img_name_to_int(paths_face_images[0])
                    last_idx = sequence_start
                    for list_idx, path_face_image in enumerate(paths_face_images):
                        image_idx = _img_name_to_int(path_face_image)

                        if flow:
                            flow_file_name = path_face_image.with_suffix(".flo").name
                            path_flow_file = (
                                source_sub_dir.parent
                                / f"flow_files_{image_size}"
                                / video_name
                                / flow_file_name
                            )

                            if path_flow_file.exists():
                                if last_idx + 1 != image_idx:
                                    sequence_start = image_idx
                                elif (
                                    image_idx - sequence_start
                                    >= min_sequence_length - 1
                                ):
                                    filtered_images_idx.append(list_idx)
                                last_idx = image_idx
                        else:
                            if last_idx + 1 != image_idx:
                                sequence_start = image_idx
                            elif image_idx - sequence_start >= min_sequence_length - 1:
                                filtered_images_idx.append(list_idx)
                            last_idx = image_idx

                    # for the test-set all frames are going to be taken
                    # otherwise distribute uniformly
                    if split_name == TRAIN_NAME:
                        samples_per_video = samples_per_video_train
                    elif split_name == VAL_NAME:
                        samples_per_video = samples_per_video_val
                    elif split_name == TEST_NAME:
                        samples_per_video = samples_per_video_test

                    selected_frames = _select_frames(
                        len(filtered_images_idx), samples_per_video
                    )

                    sampled_images_idx = np.asarray(filtered_images_idx)[
                        selected_frames
                    ]
                    file_list.add_data_points(
                        paths_face_images,
                        target_label=target,
                        split=split_name,
                        sampled_images_idx=sampled_images_idx,
                    )

    file_list.save(output_file)
    logger.info(f"{output_file} created.")


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--output_dir", required=True, type=click.Path(exists=True))
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.FF_METHODS
)
@click.option(
    "--compressions",
    "-c",
    multiple=True,
    default=[Compression.c40, Compression.c23, Compression.raw],
)
@click.option("--samples_per_video_train", default=270)
@click.option("--samples_per_video_val", default=20)
@click.option("--samples_per_video_test", default=-1)
@click.option(
    "--min_sequence_length",
    default=1,
    help="Indicates how many preceeded consecutive frames make a frame eligible (i.e."
    "if set to 5 frame 0004 is eligible if frames 0000-0003 are present as well.",
)
@click.option("--aif", is_flag=True)
@click.option("--flow", is_flag=True)
@click.option("--image_size", default=224)
def create_file_list(
    source_dir_root,
    output_dir,
    methods,
    compressions,
    samples_per_video_train,
    samples_per_video_val,
    samples_per_video_test,
    min_sequence_length,
    aif,
    flow,
    image_size,
):
    if aif:
        methods = FaceForensicsDataStructure.AIF_METHODS
        compressions = [Compression.raw]
        samples_per_video_train = 0
        samples_per_video_val = 0
        samples_per_video_test = 50

    if image_size == 112:
        data_types = [DataType.face_images_tracked_112]
    elif image_size == 224:
        data_types = [DataType.face_images_tracked_224]
    else:
        raise ValueError("image_size needs to be either 112 or 224")

    flow_part = f"_flow_files_{image_size}_" if flow else "_"
    output_file = (
        "_".join([str(method) for method in methods])
        + "_"
        + "_".join([str(compression) for compression in compressions])
        + "_"
        + "_".join([str(data_type) for data_type in data_types])
        + f"{flow_part}"
        + str(samples_per_video_train)
        + "_"
        + str(samples_per_video_val)
        + "_"
        + str(samples_per_video_test)
        + "_"
        + str(min_sequence_length)
        + ".json"
    )
    output_file = Path(output_dir) / output_file

    try:
        # if file exists, we don't have to create it again
        FileList.load(output_file)
        logger.warning("Reusing already created file!")
    except FileNotFoundError:
        _create_file_list(
            source_dir_root,
            output_file,
            methods,
            compressions,
            data_types,
            samples_per_video_train,
            samples_per_video_val,
            samples_per_video_test,
            min_sequence_length,
            aif,
            flow,
            image_size,
        )

    for split in [TRAIN_NAME, VAL_NAME, TEST_NAME]:
        data_set = FileList.get_dataset_form_file(output_file, split)
        logger.info(f"{split}-data-set: {data_set}")


if __name__ == "__main__":
    create_file_list()
