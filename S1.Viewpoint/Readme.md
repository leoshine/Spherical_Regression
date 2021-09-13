## Prepare Dataset

Please read through [this instruction](../dataset/Pascal3D/prepare_data/Readme.md) on the data organisation, data structure and data preparation.

## Usage

```bash
# e.g. bash ../trainval.sh  {gpu_ids}  {remain_args}
bash trainval.sh 0,1 --net_arch=resnet101
```

This example uses 2 GPUs to train with resetnet101 backbone. A work dir is created at:

```
./snapshots/Pascal3D_with_syn/reg_Euler2D_Sexp_Net/resnet101.torchmodel/
```

Testing will be performed every certain iterations, with the test evaluation result is wrote to ```eval.cache.txt``` and test prediction is wrote to ```rslt.cache.txt```.

<br>

### Important note: 
By default, this code uses pytorch pretrained weight to initialize the network. However, we find the pretrained weight from pytorch works slightly worse that the caffe pretrained weight (which is how we generate the results in the paper). To use the caffe pretrained weight, you need to download them from [here](https://drive.google.com/drive/folders/1b0tRAyKhCzbrOiew6e07pgaOfHcHrMTB)
and put it under

```
{code_root}/pylibs/pytorch_util/pretrained_model.cache/
```

To train with caffe pretrained weight, you can start like this:

```bash
# e.g. bash ../trainval.sh  {gpu_ids}  {remain_args}
bash trainval.sh 0,1 --net_arch=resnet101  --pretrained=caffemodel
```



<br><br>

## Check result.

```bash
cat {work_dir}/out_eval_path.txt
```



