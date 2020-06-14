# Dynamic Refinement Network for Oriented and Densely packed Object Detection
Xingjia Pan, Yuqiang Ren, Kekai Sheng, Weiming Dong, Haolei Yuan, Xiaowei Guo, Chongyang Ma, Changsheng Xu

This repository is the official PyTorch implementation of paper [Dynamic Refinement Network for Oriented 
and Densely Packed Object Detection.](https://arxiv.org/pdf/2005.09973.pdf) 



## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Evaluation

We support demo for image/ image folder, video, and webcam. 

First, download the models (By default, [ctdet_coco_dla_2x](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT) for detection and 
[multi_pose_dla_3x](https://drive.google.com/open?id=1PO1Ax_GDtjiemEmDVD7oPWwqQkUu28PI) for human pose estimation) 
from the [Model zoo](readme/MODEL_ZOO.md) and put them in `CenterNet_ROOT/models/`.

For object detection on images/ video, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
~~~

You can add `--debug 2` to visualize the heatmap outputs.
You can add `--flip_test` for flip test.


## Training

## Citation

If you find this project useful for your research, please use the following BibTeX entry.
```
@article{pan2020dynamic,
  title={Dynamic Refinement Network for Oriented and Densely Packed Object Detection},
  author={Xingjia Pan and Yuqiang Ren and Kekai Sheng and Weiming Dong and Haolei Yuan and Xiaowei Guo and Chongyang Ma and Changsheng Xu},
  booktitle={CVPR},
  pages={1--8},
  year={2020}
}
```
## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.  
Xingjia Pan: xingjia.pan@nlpr.ia.ac.cn  
Yuqiang Ren: condiren@tencent.com