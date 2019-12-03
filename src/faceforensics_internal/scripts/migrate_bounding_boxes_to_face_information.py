import logging

import click

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure

logger = logging.getLogger(__file__)


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
def migrate_bounding_boxes_to_face_information(source_dir_root, methods):

    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=(
            Compression.c40,
            Compression.c23,
            Compression.raw,
            Compression.masks,
        ),
        data_types=(DataType.bounding_boxes,),
    )

    target_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=(
            Compression.c40,
            Compression.c23,
            Compression.raw,
            Compression.masks,
        ),
        data_types=(DataType.face_information,),
    )

    for source_sub_dir, target_sub_dir in zip(
        source_dir_data_structure.get_subdirs(), target_dir_data_structure.get_subdirs()
    ):

        if not source_sub_dir.exists():
            continue

        target_sub_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Processing {source_sub_dir.parts[-2]}, {source_sub_dir.parts[-3]}"
        )

        source_sub_dir.rename(target_sub_dir)


if __name__ == "__main__":
    migrate_bounding_boxes_to_face_information()
