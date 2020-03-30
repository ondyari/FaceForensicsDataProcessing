import json

import click

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", "-c", multiple=True, default=[Compression.raw])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.DFDC_METHODS
)
def create_split_json(source_dir_root, compressions, methods):
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.face_images_tracked,),
    )

    video_names = []
    for source_sub_dir in source_dir_data_structure.get_subdirs():
        for video_path in sorted(source_sub_dir.iterdir()):
            video_name = video_path.stem
            video_names.append([video_name])

    with open("train_dfdc.json", "w") as outfile:
        json.dump(video_names, outfile)


if __name__ == "__main__":
    create_split_json()
