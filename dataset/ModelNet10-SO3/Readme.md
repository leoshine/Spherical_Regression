## Download Dataset 

The dataset is pre-build as lmdb database format.

```bash
cd  Spherical_Regression/dataset/

# download dataset 
wget http://isis-data.science.uva.nl/shuai/datasets/ModelNet10-SO3.tar.gz

# unzip and overwrite ModelNet10-SO3 folder
tar xzvf ModelNet10-SO3.tar.gz

# You should find following 3 lmdb database folders extracted:
#  (1) train_100V.Rawjpg.lmdb : 
#        training set with 100 random sampled views per CAD model. 
#  (2) train_20V.Rawjpg.lmdb
#        training set with  20 random sampled views per CAD model. 
#  (3) test_20V.Rawjpg.lmdb  
#        test set with 20 random sampled views per CAD model. 

```

<br>
<br>


### Or, you can download from our googe drive: 

[https://drive.google.com/file/d/17GLZbNTDq8B_MOgrV1TiJPoqcm_oQ_mK/view?usp=sharing](https://drive.google.com/file/d/17GLZbNTDq8B_MOgrV1TiJPoqcm_oQ_mK/view?usp=sharing)

And unzip ModelNet10-SO3.zip to overwrite ModelNet10-SO3 folder.

(You may need to use a browser to download.)
