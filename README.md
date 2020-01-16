# FaceForensics Data Processing

This repository provides an introduction of all data preprocessing and dataloaders for the FaceForensics image dataset that are sharable between multiple projects

## Setup

 - Poetry: https://github.com/sdispater/poetry (`curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`): poetry install
 - pre-commit: https://pre-commit.com/ (`pip install pre-commit`): Use before committing

## Usage

- extract_compressed_videos: extracts frames from videos
- extract_face_locations: uses dlib to extract face locations from videos for each frame. Saves it as .json to disk
- migrate_bounding_boxes_to_face_information: renames bounding_box folders to face_information_folders
- extract_faces_from_bounding_boxes: uses previously saved bounding boxes to extract face images from images
- extract_mask_bounding_boxes: extract bounding boxes from mask videos (used later for checking validity of bounding boxes in videos)
- extract_faces_tracked_from_bounding_boxes: same as other face extraction script but with tracking and can use mask information
- resample_videos: resamples videos to new frame rate

- create_file_list: creates a filelist for easier sharing and comparing of datasets

## Download

Using the above commands the FF++ dataset was preprocessed and made available via this link: http://kaldir.vc.in.tum.de/ff_all_jsons.zip
