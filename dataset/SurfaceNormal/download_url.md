## **Note & Warning**: 
The data preparation mainly following [Zhang et al. CVPR17](https://github.com/yindaz/surface_normal#data).
However, by the moment I wrote this document, I find the links they hosted are not working any more. Please consider to contact corresponding authors if you need the original release. 

However, for the sake of reproducing results in our paper,
I provide you here the pre-processed version of both NYUv2 and pbrs datasets (which are in lmdb format).

## Download our pre-processed database

```bash
# nyu_v2 dataset lmdb (2.0 GB)
wget http://isis-data.science.uva.nl/shuai/datasets/SurfaceNormal/nyu_v2.tar.gz
tar xzvf nyu_v2.tar.gz

# pbrs synthetic dataset lmdb (46GB)
wget http://isis-data.science.uva.nl/shuai/datasets/SurfaceNormal/pbrs.tar.gz
tar xzvf pbrs.tar.gz

```



<br>
<br>
<br>


### Reference
[Zhang et al. CVPR17] Physically-based rendering for indoor scene understanding using convolutional neural networks.



<!-- 

## NYU v2

For details, please check [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).


```bash
mkdir ./download.cache

# Training data list
wget http://pbrs.cs.princeton.edu/pbrs_release/nyu/trainNdxs.txt  -P download.cache/
# Testing data list
wget http://pbrs.cs.princeton.edu/pbrs_release/nyu/testNdxs.txt   -P download.cache/
# Color image and ground truth.
wget http://pbrs.cs.princeton.edu/pbrs_release/nyu/nyu_data.zip   -P download.cache/
```



## Synthetic dataset (pbrs).

For details, please check this paper:

*Yinda Zhang, Shuran Song, Ersin Yumer, Manolis Savva, Joon-Young Lee, Hailin Jin, and Thomas Funkhouser.* 
**Physically-based rendering for indoor scene understanding using convolutional neural networks.** 
In CVPR, 2017.

and their github page:
[https://github.com/yindaz/surface_normal
](https://github.com/yindaz/surface_normal
)



### Download.





```bash
# Data list
wget http://pbrs.cs.princeton.edu/pbrs_release/data/data_goodlist_v2.txt   -P download.cache/
# Surface normal ground truth (27GB)
wget http://pbrs.cs.princeton.edu/pbrs_release/data/normal_v2.zip          -P download.cache/
# Color image (278GB)
wget http://pbrs.cs.princeton.edu/pbrs_release/data/mlt_v2.zip             -P download.cache/

```





To train on synthetic image, you can find the training data from http://pbrs.cs.princeton.edu. Specifically,

Color image: http://pbrs.cs.princeton.edu/pbrs_release/data/mlt_v2.zip (278GB)
Surface normal ground truth: http://pbrs.cs.princeton.edu/pbrs_release/data/normal_v2.zip (27GB)
Data list: http://pbrs.cs.princeton.edu/pbrs_release/data/data_goodlist_v2.txt
To experiment on NYUv2 data,

Color image and ground truth: http://pbrs.cs.princeton.edu/pbrs_release/nyu/nyu_data.zip. This file is converted using data from http://www.cs.nyu.edu/~deigen/dnl/ and http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
Training data list: http://pbrs.cs.princeton.edu/pbrs_release/nyu/trainNdxs.txt
Testing data list: http://pbrs.cs.princeton.edu/pbrs_release/nyu/testNdxs.txt

comments -->
