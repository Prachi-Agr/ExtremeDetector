# ExtremeDetector
We provide code for low visibility object detection framework - Extreme Detector.
1. Run framework.py to train CNN to learn degradation present in image.
2. Run generate_enhanced.py to enhance images based on the degrdation predicted by trained CNN. Accordingly pass the images to image enhancement models.
3. model_checkpoint contains pre-trained CNN on human action recognition dataset. You can find the dataset here. https://drive.google.com/drive/u/0/folders/1PmxiF1z1UKbDSz1GaA0xsVqkEwjPEFa8
4. Use the generated enhanced images to train an object detection model. We use YOLOv5 in our experiments. 

To generate the low visbility images dataset, we use [FoHIS](https://github.com/noahzn/FoHIS) for adding fog effect and [Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection](https://github.com/cuiziteng/ICCV_MAET) for adding low light effect.
To generate the snowfall effect, we provide the script in synthetic_snowfall.ipynb

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

