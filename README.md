# Hyper Volume Generative Adversarial Network - hGAN

Replication of [Stabilizing GAN Training with Multiple Random Projections](https://arxiv.org/abs/1705.07831) and extension including training with multi-objective training via hyper volume maximization

## To run

Download the [cropped and aligned version of CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and unzip it

```
python train.py --ndiscrimiators 12
```

```
optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --epochs N            number of epochs to train (default: 500)
  --lr LR               learning rate (default: 0.0002)
  --beta1 lambda        Adam beta param (default: 0.5)
  --beta2 lambda        Adam beta param (default: 0.99)
  --ndiscriminators NDISCRIMINATORS
                        Number of discriminators. Default=8
  --checkpoint-epoch N  epoch to load for checkpointing. If None, training
                        starts from scratch
  --checkpoint-path Path
                        Path for checkpointing
  --data-path Path      Path to data
  --workers WORKERS     number of data loading workers
  --seed S              random seed (default: 1)
  --save-every N        how many epochs to wait before logging training
                        status. Default is 1
  --hyper-mode          enables training with hypervolume maximization
  --nadir nadir         Nadir point for the case of hypervolume maximization
                        (default: 1.1)
  --no-cuda             Disables GPU use
```

## Tested with


- Python 3.6
- Pytorch 0.3.0

## To do

- Scheduler for the nadir point

Collaborators: Isabela Albuquerque, Breandan Considine
