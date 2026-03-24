# NSMLIE

The Pytorch Implementation of NSMLIE. 

## Introduction
In this project, we use Ubuntu 16.04.5, Python 3.7, Pytorch 1.12.0 and one NVIDIA RTX 2080Ti GPU. 

## Datasets
Training and testing dataset are available at [Google Drive](https://drive.google.com/drive/folders/1J62xaQh1fTXEYFp53b5l5Mxli0UdeetJ?usp=sharing).

### Testing

The pretrained model is in the ./weights.

Check the model and image pathes in test.py, and then run:

```
python test.py
```

### Training

To train the model, you need to prepare our training dataset.

Check the dataset path in train.py, and then run:
```
python train.py
```

## Citation

If you find PairLIE is useful in your research, please cite our paper:

