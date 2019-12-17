import logging
import multiprocessing as mp
import os
import subprocess

import click
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure

logger = logging.getLogger(__file__)


def _resampled_video(args):
    video, resampled_video_folder, fps = args
    try:
        subprocess.check_call(
            f"/home/sebastian/bin/ffmpeg -i {video} -c:v libx264rgb -crf 0 -c:a aac "
            f"-filter:v fps=fps={fps} {resampled_video_folder/video.name} -y",
            stdout=open(os.devnull, "wb"),
            stderr=open(os.devnull, "wb"),
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"Some issue with {video}")
        logger.info(str(e))


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", "-c", multiple=True, default=[Compression.c40])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
@click.option("--fps", default=25.0)
def resample_videos(source_dir_root, compressions, methods, fps):
    videos_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.videos,),
        methods=methods,
    )

    resampled_videos_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        methods=methods,
        data_types=(DataType.resampled_videos,),
    )

    for videos, resampled_videos in zip(
        videos_data_structure.get_subdirs(),
        resampled_videos_data_structure.get_subdirs(),
    ):
        logger.info(f"Current method: {videos.parents[1].name}")

        resampled_videos.mkdir(exist_ok=True)

        p = mp.Pool(mp.cpu_count())

        p.map(
            _resampled_video,
            tqdm(
                [
                    (_video_folder, resampled_videos, fps)
                    for _video_folder in sorted(videos.iterdir())
                ]
            ),
        )


if __name__ == "__main__":
    resample_videos()
