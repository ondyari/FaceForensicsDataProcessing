import multiprocessing as mp
from pathlib import Path

import click
from joblib import delayed
from joblib import Parallel
from PIL.Image import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from tqdm import tqdm

from faceforensics_internal.utils import Compression
from faceforensics_internal.utils import DataType
from faceforensics_internal.utils import FaceForensicsDataStructure


def _resize_face_images(face_images_path: Path, target_sub_dir: Path, size: int):

    output_dir = target_sub_dir / face_images_path.name
    if output_dir.exists():
        return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    for face_image_path in face_images_path.glob("*.png"):
        try:
            face_image = default_loader(face_image_path)
        except OSError:
            print(f"OSError: {face_image_path}")
            continue

        transform = Compose([Resize(size), CenterCrop(size)])
        face_image_resized: Image = transform(face_image)

        face_image_resized_path = output_dir / face_image_path.name
        face_image_resized.save(str(face_image_resized_path))


@click.command()
@click.option("--source_dir_root", required=True, type=click.Path(exists=True))
@click.option("--target_dir_root", required=True, type=click.Path(exists=True))
@click.option("--compressions", "-c", multiple=True, default=[Compression.raw])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.DFDC_METHODS
)
@click.option("--cpu_count", default=mp.cpu_count(), type=click.INT)
@click.option("--size", default=224, type=click.INT)
def resize_face_images(
    source_dir_root, target_dir_root, compressions, methods, cpu_count, size
):
    source_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        methods=methods,
        compressions=compressions,
        data_types=(DataType.face_images_tracked,),
    )

    if size == 112:
        target_dir_data_type = DataType.face_images_tracked_112
    elif size == 224:
        target_dir_data_type = DataType.face_images_tracked_224
    else:
        raise ValueError("size can either be 112 or 224")

    target_dir_data_structure = FaceForensicsDataStructure(
        target_dir_root,
        methods=methods,
        compressions=compressions,
        data_types=(target_dir_data_type,),
    )

    for source_sub_dir, target_sub_dir in zip(
        source_dir_data_structure.get_subdirs(), target_dir_data_structure.get_subdirs()
    ):

        if not source_sub_dir.exists():
            continue

        print(f"Processing {source_sub_dir.parts[-2]}, {source_sub_dir.parts[-3]}")

        Parallel(n_jobs=cpu_count)(
            delayed(
                lambda _face_images_path: _resize_face_images(
                    _face_images_path, target_sub_dir, size
                )
            )(face_images_path)
            for face_images_path in tqdm(sorted(source_sub_dir.iterdir()))
        )


if __name__ == "__main__":
    resize_face_images()
