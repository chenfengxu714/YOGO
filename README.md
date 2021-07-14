# You Only Group Once: Efficient Point-Cloud Processing with Token Representation and Relation Inference Module (IROS 2021)
By Chenfeng Xu, Bohan Zhai, Bichen Wu, Tian Li, Wei Zhan, Peter Vajda, Kurt Keutzer, and Masayoshi Tomizuka.

<p align="center">
    <img src="./figure/intro.png"/ width="750">
</p>


This repository contains a Pytorch implementation of YOGO, a new, simple, and elegant model for point-cloud processing. The framework of our YOGO is shown below:

<p align="center">
    <img src="./figure/framework.png"/ width="750">
</p>

Selected quantitative results of different approaches on the ShapeNet and S3DIS dataset.

### ShapeNet part segmentation:

|   Method       | mIoU | Latency (ms) | GPU Memory (GB)|
| ---------------|------|--------------|----------------|
| PointNet       | 83.7 |21.4          | 1.5            |
| RSNet          | 84.9 | 73.8         | 0.8            |
| PointNet++     | 85.1 |77.7          | 2.0            |
| DGCNN          | 85.1 |86.7          | 2.4            |
| PointCNN       | 86.1 |134.2         | 2.5            |
| YOGO(KNN)      | 85.2 |25.6          | 0.9            |
|YOGO(Ball query)| 85.1 |21.3          | 1.0            |

### S3DIS scene parsing:

|   Method       | mIoU | Latency (ms) | GPU Memory (GB)|
| ---------------|------|--------------|----------------|
| PointNet       | 42.9 |24.8          | 1.0            |
| RSNet          | 51.9 |111.5         | 1.1            |
| PointNet++*    | 50.7 |501.5         | 1.6            |
| DGCNN          | 47.9 |174.3         | 2.4            |
| PointCNN       | 57.2 |282.4         | 4.6            |
| YOGO(KNN)      | 54.0 |27.7          | 2.0            |
|YOGO(Ball query)| 53.8 |24.0          | 2.0            |

For more detail, please refer to our paper: [YOGO](https://arxiv.org/abs/2103.09975). The work is a follow-up work to [SqueezeSegV3](https://arxiv.org/abs/2004.01803) and [Visual Transformers](https://arxiv.org/abs/2006.03677). If you find this work useful for your research, please consider citing:

```
@article{xu2021you,
  title={You Only Group Once: Efficient Point-Cloud Processing with Token Representation and Relation Inference Module},
  author={Xu, Chenfeng and Zhai, Bohan and Wu, Bichen and Li, Tian and Zhan, Wei and Vajda, Peter and Keutzer, Kurt and Tomizuka, Masayoshi},
  journal={arXiv preprint arXiv:2103.09975},
  year={2021}
}
```

Related works:
```
@inproceedings{xu2020squeezesegv3,
  title={Squeezesegv3: Spatially-adaptive convolution for efficient point-cloud segmentation},
  author={Xu, Chenfeng and Wu, Bichen and Wang, Zining and Zhan, Wei and Vajda, Peter and Keutzer, Kurt and Tomizuka, Masayoshi},
  booktitle={European Conference on Computer Vision},
  pages={1--19},
  year={2020},
  organization={Springer}
}
```

```
@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
**YOGO** is released under the BSD license (See [LICENSE](https://github.com/chenfengxu714/YOGO/blob/master/LICENSE) for details).

## Installation

The instructions are tested on Ubuntu 16.04 with python 3.6 and Pytorch 1.5 with GPU support.

* Clone the YOGO repository:


```shell
git clone https://github.com/chenfengxu714/YOGO.git
```

* Use pip to install required Python packages:

```shell
pip install -r requirements.txt
```

* Install KNN library:

```shell
cd convpoint/knn/
python setup.py install --home='.'
```

* Click to download [ShapeNet](https://www.shapenet.org) and [S3DIS](https://web.archive.org/web/20200707221857/http://buildingparser.stanford.edu/dataset.html#Download) dataset.


## Prepare the S3DIS dataset
```
cd data/s3dis
python prepare_data.py
```

## Pre-trained Models
The pre-trained YOGO is avalible at [Google Drive](https://drive.google.com/drive/folders/1LDUNG-K9xTCX6TF2tI71vAnmZkjDf6NH?usp=sharing), you can directly download them.

## Inference
To infer the predictions for the entire dataset:

```shell
python train.py [config-file] --devices [gpu-ids] --evaluate --configs.evaluate.best_checkpoint_path [path to the model checkpoint]
```

for example, you can run the below command for ShapeNet inference:

```shell
python train.py configs/shapenet/yogo/yogo.py --devices 0 --evaluate --configs.evaluate.best_checkpoint_path ./runs/shapenet/best.pth
```


## Training:
To train the model:

```shell
python train.py [config-file] --devices [gpu-ids] --evaluate --configs.evaluate.best_checkpoint_path [path to the model checkpoint]
```

for example, you can run the below command for ShapeNet training:

```shell
python train.py configs/shapenet/yogo/yogo.py --devices 0
```

You can run the below command for multi-gpu training:

```shell
python train.py configs/shapenet/yogo/yogo.py --devices 0,1,2,3
```

Note that we conduct training on Titan RTX gpu, you can modify the batch size according your GPU memory, the performance is slightly different.

## Acknowledgement:
The code is modified from [PVCNN](https://github.com/mit-han-lab/pvcnn) and the code for KNN is from [Pointconv](https://github.com/aboulch/ConvPoint/tree/master/convpoint).

