# RoadMarkingExtraction
This repository contains a C++ implementation of the automatic extraction, classification and vectorization of road markings from MLS point cloud. It's based on traditional image & point cloud processing approaches, which can act as the easy baseline for novel learning-based methods.

![alt text](demo/teaser.gif)

-------

## About
Version 1.2. The current build is also compatible with Ubuntu 22.04, PCL 1.12 and OpenCV 4.5.

Dependencies: CMake 3.15 or later, a C++14 compiler, Eigen3, PCL, OpenCV, Boost and libLAS. DXFLib is included in the repository.

Application Scenarios: MLS or ALS point cloud for highway or urban roads

## How to use
1. Install the development packages for Eigen3, PCL, OpenCV and Boost. On Ubuntu 22.04:

```
sudo apt install build-essential cmake git libeigen3-dev libpcl-dev libopencv-dev libboost-all-dev
```

Install libLAS separately if it is unavailable in the system. For example, to build it locally without modifying system libraries:

```
git clone https://github.com/libLAS/libLAS.git /tmp/roadmarking-liblas
git -C /tmp/roadmarking-liblas checkout 0839c6cab5fd8b629cc8e5e298193fa56078bccb
cmake -S /tmp/roadmarking-liblas -B /tmp/roadmarking-liblas-build \
  -DCMAKE_INSTALL_PREFIX="$PWD/.deps" -DWITH_GDAL=OFF -DWITH_GEOTIFF=OFF
cmake --build /tmp/roadmarking-liblas-build -j2
cmake --install /tmp/roadmarking-liblas-build
```

Run these commands from the repository root. This configuration supports uncompressed LAS, but not LAZ compression or libLAS coordinate reference system transformations.

2. Build the repository:

```
cmake -S . -B build
cmake --build build -j1
```

3. Prepare the model pool (take `./model_pool/xxx` as an example) and configure the parameter list (take `./config/xxx.txt` as an example).

4. Configure the input (`*.las` or `*.pcd` point cloud) and output path in `./script/run_xxx.sh`.

5. Run `sh ./script/run_xxx.sh`. 

6. Check the results in your output folder. You may use CloudCompare to visualize the point cloud and use AutoCAD or [ShareCAD](https://beta.sharecad.org/) to visualize the dxf files.

------
### Citation

If you find this code useful for your work or use it in your project, please consider citing:

```
@article{pan2019automatic,
  title={Automatic Road Markings Extraction, Classification and Vectorization from Mobile Laser Scanning Data.},
  author={Pan, Yue and Yang, B and Li, S and Yang, H and Dong, Z and Yang, X},
  journal={International Archives of the Photogrammetry, Remote Sensing \& Spatial Information Sciences},
  year={2019}
}
```

[paper link](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2-W13/1089/2019/)

### Workflow
 ![alt text](demo/framework.png)
### Image Processing
 ![alt text](demo/image_process.png)
### Demo
 ![alt text](demo/scenarios.png)
 
### Acknowledgement:
Thanks [hibetterheyj](https://github.com/hibetterheyj) for the effort to migrate the codes on Linux.
