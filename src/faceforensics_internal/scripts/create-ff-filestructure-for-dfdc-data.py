import json
import logging
import shutil
from pathlib import Path
from pprint import pformat

import click
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--dfdc_root_dir", type=click.Path())
def main(**kwargs):
    logger.info(f"kwargs:\n{pformat(kwargs)}")

    dfdc_root_dir = Path(kwargs["dfdc_root_dir"])
    face_images_dir = dfdc_root_dir / "face_images"
    parts_dir = dfdc_root_dir / "parts"

    dfdc_ff_dir = dfdc_root_dir / "dfdc_ff"
    manipulated_dir = (
        dfdc_ff_dir / "manipulated_sequences" / "dfdc-fake" / "raw" / "face_images"
    )
    original_dir = (
        dfdc_ff_dir / "original_sequences" / "dfdc-real" / "raw" / "face_images"
    )

    # Iterate over parts
    sorted_parts_dirs = sorted(parts_dir.iterdir())
    parts_pbar = tqdm(
        enumerate(sorted_parts_dirs),
        desc="Processing parts",
        total=len(sorted_parts_dirs),
        position=0,
    )
    for part_num, part_dir in parts_pbar:
        # Load metadata
        with open(str(part_dir / "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Iterate over videos
        video_pbar = tqdm(
            sorted(metadata.items()), desc="Processing videos", position=1
        )
        for video_name, video_metadata in video_pbar:
            video_name = Path(video_name).stem
            face_image_dir = face_images_dir / video_name

            if face_image_dir.exists():
                label = video_metadata["label"]

                if label == "REAL":
                    shutil.move(str(face_image_dir), str(original_dir / video_name))
                elif label == "FAKE":
                    shutil.move(str(face_image_dir), str(manipulated_dir / video_name))
            else:
                logger.warning(f"{face_image_dir} does not exist")


if __name__ == "__main__":
    main()
