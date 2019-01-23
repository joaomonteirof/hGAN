# Hyper Volume Generative Adversarial Network - hGAN

Training the Generator with multi-objective training via hyper volume maximization

## To run

### Data

MNIST and Cifar-10 will be downloaded automatically

Stacked MMNIST has to be built and dumped into an .hdf file prior to training. Download the data from [https://ufile.io/k854s](https://ufile.io/k854s), or build it by running:

```
python gen_data.py --data-size 50000 --data-path /path/to/download/mnist --out-file /out/file/path
```

CelebA: Download the [cropped and aligned version of CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and unzip it

Cats dataset can be downloaded from: [https://ufile.io/u6i98](https://ufile.io/u6i98)



### Training generators

cd to the the folder corresponding to the desired dataset and run:

```
python train.py --help
```

to get the list of arguments.

Example for CelebA:

```
optional arguments:
  -h, --help            Show this help message and exit
  --batch-size N        Input batch size for training (default: 64)
  --epochs N            Number of epochs to train (default: 50)
  --lr LR               Learning rate (default: 0.0002)
  --beta1 lambda        Adam beta param (default: 0.5)
  --beta2 lambda        Adam beta param (default: 0.999)
  --ndiscriminators NDISCRIMINATORS
                        Number of discriminators. Default=8
  --checkpoint-epoch N  Epoch to load for checkpointing. If None, training
                        starts from scratch
  --checkpoint-path Path
                        Path for checkpointing
  --data-path Path      Path to data
  --workers WORKERS     Number of data loading workers
  --seed S              Random seed (default: 1)
  --save-every N        How many epochs to wait before logging training
                        status. Default is 3
  --train-mode {vanilla,hyper,gman,gman_grad,loss_delta,mgd}
                        Select train mode. Default is vanilla (simple average
                        of Ds losses)
  --disc-mode {RP,MD}   Multiple identical Ds with random projections (RP) or
                        Multiple distinct Ds without projection (MD)
  --nadir-slack nadir   Factor for nadir-point update. Only used in hyper mode
                        (default: 1.5)
  --alpha alpha         Used in GMAN and loss_del modes (default: 0.8)
  --no-cuda             Disables GPU use
```

### Required args

For both cifar10_32 and 64, --fid-model-path has to be specified to allow for FID computation at train time. Download [the model used in our experiments](https://ufile.io/5ky3g)


## Computing FID with a pre-trained model 

Download the pretrained classifier for evaluation at [https://ufile.io/8udto](https://ufile.io/8udto) 
 
Example for calculating FID using a ResNet for a generator trained on CIFAR-10:
 
cd to common and run

```
python compute_FID.py --model-path Path --data-stat-path Path --data-path Path --fid-model-path Path --model-cifar resnet --nsamples 100000 --ntests 10 --dataset cifar10
```

## Computing coverage for Stacked MNIST


cd to stacked_mnist, download the pretrained classifier for evaluation at [https://ufile.io/8udto](https://ufile.io/8udto) (same as for FID). 
First, compute coverage for real data (for calculating the KL divergence), by running:

```
python coverage_real_data.py --classifier-path Path --data-path Path --out-file Path
```

Example:

```
python coverage.py --cp-folder Path --classifier-path Path --data-stat-path Path --out-file Path --n-tests 10 --n-samples 10000
```

## TensorFlow implementation for computing Inception Score and FID using author's original implementation

Inception Score: [https://github.com/openai/improved-gan/blob/master/inception_score/model.py](https://github.com/openai/improved-gan/blob/master/inception_score/model.py)

FID: [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)

## Tested with

- Python 3.6
- Pytorch > 0.4.1
