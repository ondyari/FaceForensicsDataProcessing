# FaceForensics

FaceForensics is a video dataset consisting of more than 500,000 frames containing faces from 1004 videos that can be used to study image or video forgeries. To create these videos we use an automatated version of the state of the art Face2Face approach. All videos are downloaded from Youtube and are cut down to short continuous clips that contain mostly frontal faces. In particular, we offer two versions of our dataset: 
*source-to-target*, where we reenact over 1000 videos with new facial expressions extracted from other videos, which e.g. can be used to train a classifier to detect fake images or videos,
and *self-reenactment*, where we use Face2Face to reenact the facial expressions of videos with their own facial expressions as input to get pairs of videos, which e.g. can be used to train supervised generative refinement models.
Consult our [paper](https://arxiv.org/abs/1803.09179) for more information or watch our [youtube video](https://www.youtube.com/watch?v=Tle7YaPkO_k).

## FaceForensics Data

If you would like to download the FaceForensics data, please fill out an agreement to the [FaceForensics Terms of Use](http://kaldir.vc.in.tum.de/FaceForensics/webpage/FaceForensics_TOS.pdf) and send it to us at [faceforensics@googlegroups.com](mailto:faceforensics@googlegroups.com).

If you have not received a response within a week, it is likely that your email is bouncing - please check this before sending repeat requests.

### Download

Use the download-FaceForensics.py script to download the dataset you are interested in.
You can choose to download H.264 lossless compressed or raw videos. Note that the dataset size
is ~130gb and ~3,5tb respectively.

**source-to-target:** python download.py -d \<'compressed' or 'raw'> \<output folder>   
**self-reenactment:** python download.py -d \<'selfreenactment_compressed' or 'selfreenactment_raw'> \<output folder>  
**cropped self-reenactment images:** python download.py -d selfreenactment_images' \<output folder>  
**only original videos:** python download.py -d original_videos \<output filename>

For more information, e.g. how to download only a specific part of the dataset, use the '-h' option.

### Data Organization

**Folder structure**  
We provide the split we used for the final numbers in our paper which divides the 1004 videos in
704 videos for train, 150 videos for validation and 150 videos for test, where we made sure that 
all videos come from different original video sources.
```shell
FaceForensic dataset
|-- test
    |-- altered
    Contains videos that were altered by Face2Face
    |-- mask
    Contains the masks video that were produced by the Face2Face algorithm
    |-- original
    Contains the original videos
|-- train
    -- same as test --
|-- val
    -- same as test -- 
```

**Video format**  
All videos have a constant frame rate of 30 fps and are either written raw with the 'DIB ' codec 
or have been lossless compressed with H.264. All videos are at least 480p, at most 1080 and at least
300 frames long.

**File format template**  
All video files are formatted as follows  
```
&lt;target youtube id&gt;_&lt;number&gt;_&lt;source youtube id&gt;_&lt;number&gt;.&lt;file extension&gt;
```

where a youtube id is always 11 characters long.

**Stats**  
For more information about the origin of our original videos as well as other stats about
 the dataset please consult files in the stats folder.


## Citation
If you use the FaceForensics data or code please cite:
```
@article{roessler2018faceforensics,
	author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
	title = {FaceForensics: A Large-scale Video Dataset for Forgery Detection in Human Faces},
	journal={arXiv},
	year={2018}
}
```

## Help
If you have any questions, please contact us at faceforensics@googlegroups.com


## Changelog

## License
The data is released under the [FaceForensics Terms of Use](http://kaldir.vc.in.tum.de/FaceForensics/webpage/FaceForensics_TOS.pdf), and the code is released under the MIT license.

Copyright (c) 2018