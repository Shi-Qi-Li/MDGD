# Matching Distance and Geometric Distribution Aided Learning Multiview Point Cloud Registration

This is an official implementation of [*Matching Distance and Geometric Distribution Aided Learning Multiview Point Cloud Registration*](https://ieeexplore.ieee.org/document/10669215) that is accepted to IEEE Robotics and Automation Letters.

## Abstract
Multiview point cloud registration plays a crucial role in robotics, automation, and computer vision fields. This paper concentrates on pose graph construction and motion synchronization within multiview registration. Previous methods for pose graph construction often pruned fully connected graphs or constructed sparse graph using global features aggregated from local descriptor, which may not consistently yield reliable results. To identify dependable pairs for pose graph construction, we design a network model that extracts information from the matching distance between point cloud pairs. For motion synchronization, we propose another neural network model to calculate the absolute pose in a data-driven manner, rather than optimizing inaccurate handcrafted loss functions. Our model takes into account geometric distribution information and employs a modified attention mechanism to facilitate flexible and reliable feature interaction. Experimental results on diverse indoor and outdoor datasets confirm the effectiveness and generalizability of our approach.

## Installation
First, create the conda environment.
```
conda create -n mdgd python=3.8
conda activate mdgd
pip install -r requirements.txt
```
Then, install the [knn_search](model/knn_search/README.md) and [graph_ops](model/graph_ops/README.md) in `./model`.

## Data Preparation
The data can be found from [SGHR](https://github.com/WHU-USI3DV/SGHR).

Please organize the data to `./data` following the example data structure as:
```
data/
├── 3dmatch/
    └── kitchen/
        ├── PointCloud/
            ├── cloud_bin_0.ply
            ├── gt.log
            └── gt.info
        ├── yoho_desc/
            └── 0.npy
        └── Keypoints/
            └── cloud_bin_0Keypoints.txt
├── eth/
├── scannet/
├── train/
├── val/
├── 3dmatch.pkl
├── eth.pkl
├── scannet.pkl
├── train.pkl
└── val.pkl
```
Then generate the training set by:
```
python tools/cache_data.py
```
This step creates the `cache` folder in the `./data` directory.

We provide the pre-trained model checkpoints in release page, download and put the weight files to `./ckpt` folder.

## Train
First, train the overlap estimation module by:
```
python train_iter.py --config config/train_distance.yaml
```
Then, load the overlap module weight and train the whole model by:
```
python train_iter.py --config config/train_motion.yaml --overlap ckpt/dis.pth
```

## Test
```
python test.py --config config/3dmatch.yaml --ckpt ckpt/mdgd.pth
python test.py --config config/scannet.yaml --ckpt ckpt/mdgd.pth
python test.py --config config/eth.yaml --ckpt ckpt/mdgd.pth
```

## Cite
If you find this code useful for your work, please consider citing:
```
@article{li2024matching,
  title={Matching Distance and Geometric Distribution Aided Learning Multiview Point Cloud Registration},
  author={Li, Shiqi and Zhu, Jihua and Xie, Yifan and Hu, Naiwen and Wang, Di},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgement
We thank the authors of the [LMVR](https://github.com/zgojcic/3D_multiview_reg), [MultiReg](https://github.com/yewzijian/MultiReg), [SGHR](https://github.com/WHU-USI3DV/SGHR) for open sourcing their codes.