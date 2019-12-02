import logging
import multiprocessing as mp
import subprocess
from pathlib import Path

import click
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure

logger = logging.getLogger(__file__)


def _resampled_video(video: Path, resampled_video_folder: Path):
    try:
        subprocess.check_output(
            f"/home/sebastian/bin/ffmpeg -i {video} -c:v h264 -crf 0 -c:a aac "
            f"-filter:v fps=fps=25 {resampled_video_folder/video.name}",
            stderr=subprocess.STDOUT,
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"Some issue with {video}")
        logger.info(str(e))


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", "-c", default=[Compression.c40])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
def resample_videos(source_dir_root, compressions, methods):
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

        # extract faces from videos in parallel
        Parallel(n_jobs=mp.cpu_count())(
            delayed(
                lambda _video_folder: _resampled_video(_video_folder, resampled_videos)
            )(video_folder)
            for video_folder in tqdm(sorted(videos.iterdir()))
        )


if __name__ == "__main__":
    resample_videos()
