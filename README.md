# Introduction

This project is forked from [RDS-SLAM](https://github.com/yubaoliu/RDS-SLAM). 

This project balances performance across low-dynamic and high-dynamic sequences by augmenting with a geometric probability estimate derived from the computed chi-squared error from pose optimization. 

This project also replaces MaskRCNN with higher performing LRASPP in terms of speed and accuracy, lowering the semantic delay.

# Develop environments

-   ubuntu: 18.04
-   ROS melodic
-   cuda: 11.1
-   OpenCV 4.2.0
-   Torch 1.8.0+cu111
-   Python 3.7 

# How to use

## How to build

catkin_build_ws is a ROS workspace.

[Note] You can set up a catkin workspace similarly following [this ROS tutorial](https://industrial-training-master.readthedocs.io/en/melodic/_source/session1/Create-Catkin-Workspace.html)

1. Modify the ```mrcnn_root``` variable in the script ```MaskRCNN_ROS/script/action_server.py``` to your local setup.

2. Execute the following commands.

```sh
cd ~/catkin_build_ws/src/SLAM/
./build_thirdparty.sh

cd ~/catkin_build_ws
catkin_make
```

## Set up TUM RGB-D Dataset
1. Download TUM RGB-D Dataset sequence folders from Category:Dynamic Objects (i.e., the eight sequences prepended with "fr3").
Download from: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download 

2. Save sequence folders into the folder ```/root/Dataset/TUM/freiburg3/```

## How to run demo

```sh
roslaunch segnet_ros action_server_dl.launch
```

[Important] Please run w/static to initialize **GPU** before you evaluate any datasets.

```sh
roslaunch rds_slam tum_maskrcnn_walk_static.launch
```

```sh
roslaunch rds_slam tum_maskrcnn_walk_xyz.launch
```

## How to run evaluations after demo

From catkin workspace directory, run: 
```sh
python2.7 src/RDS-SLAM/SLAM/evaluation/evaluate_rpe.py /root/Dataset/TUM/freiburg3/rgbd_dataset_freiburg3_sitting_static/groundtruth.txt /root/.ros/CameraTrajectory.txt --verbose
```

# Notes

-   [**Important**] please run a dataset to initialize the GPU  before you evaluate the time and tracking performance.
-   Please use the data listed in the original paper because the tracking performance and real-time performance is somehow related to the GPU and CPU configuration
-   The real-time performance and tracking performance can be trade off by controlling the frame rate by adjusting some  parameters in the SLAM main loop.

# References

[1] Y. Liu and J. Miura, "RDS-SLAM: Real-Time Dynamic SLAM Using Semantic Segmentation Methods," in IEEE Access, vol. 9, pp. 23772-23785, 2021, doi: 10.1109/ACCESS.2021.3050617. [PDF](https://ieeexplore.ieee.org/document/9318990)

[2] Y. Liu and J. Miura, “RDMO-SLAM: Real-Time Visual SLAM for Dynamic Environments Using Semantic Label Prediction With Optical Flow,” IEEE Access, vol. 9, pp. 106981–106997, 2021, doi: 10.1109/ACCESS.2021.3100426. [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9497091)

[3] Y. Liu and J. Miura, "KMOP-vSLAM: Dynamic Visual SLAM for RGB-D Cameras using K-means and OpenPose," 2021 IEEE/SICE International Symposium on System Integration (SII), 2021, pp. 415-420, doi: 10.1109/IEEECONF49454.2021.9382724. [PDF](https://ieeexplore.ieee.org/document/9382724)

[4] Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495. [PDF](https://arxiv.org/abs/1511.00561)

[5] Campos, Carlos, et al. "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial, and Multimap SLAM." IEEE Transactions on Robotics (2021). [PDF](https://arxiv.org/pdf/2007.11898.pdf)

[6] He, Kaiming, et al. "Mask r-cnn." Proceedings of the IEEE international conference on computer vision. 2017. [PDF](https://arxiv.org/pdf/1703.06870.pdf)

# License

-   ORB-SLAM3
    ORB-SLAM3 is released under GPLv3 license. For a list of all code/library dependencies (and associated licenses), please see [Dependencies.md](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Dependencies.md).

- [Pangolin](https://github.com/stevenlovegrove/Pangolin.git) MIT License
- [G2O](https://github.com/RainerKuemmerle/g2o.git)
- [DBoW2](https://github.com/dorian3d/DBoW2.git)
-   SegNet_ROS: 
    A ROS version of SegNet.  SegNet_ROS is released under GPLV3.
    We used [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial).

-   RDS-SLAM: 
    RDS-SLAM is released under GPLv3 license. The code/library dependencies is the same as ORB_SLAM3.

