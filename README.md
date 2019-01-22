# Hyper Volume Generative Adversarial Network - hGAN

Training the Generator with multi-objective training via hyper volume maximization

## To run

### Data

MNIST and Cifar-10 will be downloaded automatically

Stacked MMNIST has to be built and dumped into an .hdf file prior to training. Download the data from [](https://ufile.io/k854s), or build it by running:

```
python gen_data.py --data-size 50000 --data-path /path/to/download/mnist --out-file /out/file/path
```

CelebA: Download the [cropped and aligned version of CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and unzip it

Cats dataset can be downloaded from: [](https://ufile.io/u6i98)



### Training generators

cd to the the folder corresponding to the desired dataset and run:

```
python train.py --help
```

to get the list of arguments.

Example for CelebA:

```
optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --epochs N            number of epochs to train (default: 50)
  --lr LR               learning rate (default: 0.0002)
  --beta1 lambda        Adam beta param (default: 0.5)
  --beta2 lambda        Adam beta param (default: 0.999)
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
                        status. Default is 3
  --train-mode {vanilla,hyper,gman,gman_grad,loss_delta,mgd}
                        Salect train mode. Default is vanilla (simple average
                        of Ds losses)
  --disc-mode {RP,MD}   Multiple identical Ds with random projections (RP) or
                        Multiple distinct Ds without projection (MD)
  --nadir-slack nadir   factor for nadir-point update. Only used in hyper mode
                        (default: 1.5)
  --alpha alhpa         Used in GMAN and loss_del modes (default: 0.8)
  --no-cuda             Disables GPU use
```

### Testing generators

### Required args

For both cifar10_32 and 64, --fid-model-path has to be specified to allow for FID computation at train time. Download [the model used in our experiments](https://ufile.io/5ky3g)


## Tested with

- Python 3.6
- Pytorch > 0.4.1
