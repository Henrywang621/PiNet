<h1 style="color: blue;">Feature Specific Progressive Improvement for Salient Object Detection</h1>

**The authors**: Xianheng Wang, Zhaobin Liu, Veronica Liesaputra, Zhiyi Huang
<br/><br/>

The manuscript was originally submitted to Pattern Recognition on June 10, 2021, and was finally accepted on September 27, 2023. The code is currently being organized and will be uploaded gradually.
***
<br/><br/>
## Requirements

[Python 3.9.7](https://www.tensorflow.org/)

[Tensorflow 2.7.0](https://www.tensorflow.org/)

[Numpy 1.22.4](https://numpy.org/)

[OpenCV-Python 4.5.4](https://pypi.org/project/opencv-python/)

<br/><br/>
## Introduction

The overall architecture of the proposed PiNet is shown below. 
The main rationales behind our design are *level-specific feature extraction* and *progressive refinement of saliency*. The paper could be found in [Link1 (PR)](https://www.sciencedirect.com/science/article/pii/S0031320323007823) or [Link2 (Researchgate)](https://www.researchgate.net/publication/375073017_Feature_Specific_Progressive_Improvement_for_Salient_Object_Detection/citations).

<br/><br/>

![PiNet](https://github.com/Henrywang621/PiNet/assets/49090477/00fa14e5-64f5-4096-8a89-8a225821dc8d)

<br/><br/>

## Saliency maps
**PiNet-V** (VGG-16 as the backbone): [Google Drive](https://drive.google.com/drive/folders/1gkTOObZrFnId_iSFLRVvvvwRpkaHFreB?usp=sharing)

**PiNet-R** (ResNet-50 as the backbone): [Google Drive](https://drive.google.com/drive/folders/1mG_JZn1oh8JU-cor3ElCvi76aTlgDyZB?usp=sharing )

**PiNet-B3** (EfficientNet-B3 as the backbone): [Google Drive](https://drive.google.com/drive/folders/1obznKaEfyH0xFB04EuYvDJAZ7lJqOdjr?usp=sharing)

**PiNet-B4** (EfficientNet-B4 as the backbone): [Google Drive](https://drive.google.com/drive/folders/1lDTBRg3hXm9TanjocHo2dq1sGWTdvsHH?usp=sharing)

<br/><br/>

## Quantitative comparisons with SOTA models

![image](https://github.com/Henrywang621/PiNet/assets/49090477/4348fe57-a464-42d0-8537-78bfc60fecd4)

<br/><br/>

![image](https://github.com/Henrywang621/PiNet/assets/49090477/2ef827f4-8fcb-4513-bf05-63ae6768b45a)

<br/><br/>

## Citation

Please cite this work if it is helpful.

```@article{WANG2024110085,
title = {Feature specific progressive improvement for salient object detection},
journal = {Pattern Recognition},
volume = {147},
pages = {110085},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.110085},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323007823},
author = {Xianheng Wang and Zhaobin Liu and Veronica Liesaputra and Zhiyi Huang},
keywords = {Salient object detection, Fully convolutional neural network, Level-specific feature extraction, Progressive refinement of saliency},
}

