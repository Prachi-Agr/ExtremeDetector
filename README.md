# ExtremeDetector
We provide code for low visibility object detection framework - Extreme Detector.
1. Run framework.py to train CNN to learn degradation present in image.
2. Run generate_enhanced.py to enhance images based on the degrdation predicted by trained CNN. Accordingly pass the images to image enhancement models.
3. model_checkpoint contains pre-trained CNN on human action recognition dataset. You can find the dataset here. https://drive.google.com/drive/u/0/folders/1PmxiF1z1UKbDSz1GaA0xsVqkEwjPEFa8
4. Use the generated enhanced images to train an object detection model. We use YOLOv5 in our experiments. 

To generate the low visbility images dataset, we use [FoHIS](https://github.com/noahzn/FoHIS) [1] for adding fog effect and [Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection](https://github.com/cuiziteng/ICCV_MAET) [2] for adding low light effect.
To generate the snowfall effect, we provide the script in synthetic_snowfall.ipynb

Our enhancement pool consists of pre-trained models for fog removal, low light removal and snowfall removal. We use Zero-DCE [4] for low light image enhancement, FFA-Net [5] for removing fog and Deep Detailed Network [3] for removing snow.


## References
<a id="1">[1]</a> 
@inproceedings{zhang2017towards,
title={Towards simulating foggy and hazy images and evaluating their authenticity},
author={Zhang, Ning and Zhang, Lin and Cheng, Zaixi},
booktitle={International Conference on Neural Information Processing},
pages={405--415},
year={2017},
organization={Springer}
}.

<a id="2">[2]</a>
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Ziteng and Qi, Guo-Jun and Gu, Lin and You, Shaodi and Zhang, Zenghui and Harada, Tatsuya},
    title     = {Multitask AET With Orthogonal Tangent Regularity for Dark Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2553-2562}
}

<a id="3">[3]</a>
@inproceedings{fu2017removing,
  title={Removing rain from single images via a deep detail network},
  author={Fu, Xueyang and Huang, Jiabin and Zeng, Delu and Huang, Yue and Ding, Xinghao and Paisley, John},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3855--3863},
  year={2017}
}

<a id="4">[4]</a>
@inproceedings{guo2020zero,
  title={Zero-reference deep curve estimation for low-light image enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1780--1789},
  year={2020}
}

<a id="5">[5]</a>
@inproceedings{qin2020ffa,
  title={FFA-Net: Feature fusion attention network for single image dehazing},
  author={Qin, Xu and Wang, Zhilin and Bai, Yuanchao and Xie, Xiaodong and Jia, Huizhu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11908--11915},
  year={2020}
}

