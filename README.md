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


### Sampling from the generator

cd to the the folder corresponding to the desired dataset and run:

```
python test.py --help
```

to get the list of arguments.

```
--cp-path Path  Checkpoint/model path
--n-tests N     Number of samples to generate (default: 4)
--no-plots      Disables plot of train/test losses
--no-cuda       Disables GPU use
```

### Evaluating generators 

Download the pretrained classifier for evaluation at [https://ufile.io/8udto](https://ufile.io/8udto) 
 
cd to common

```
python compute_FID.py --help
```

to get the list of arguments.


```
Arguments:
	--model-path Path       Checkpoint/model path
	--data-stat-path Path   Path to file containing test data statistics
	--data-path	Path        Path to data if data statistics are not provided
	--fid-model-path Path   Path to fid model
	--model-cifar {resnet,vgg,inception} 
                            Model for FID computation on CIFAR-10 (default: ResNet)
	--model-mnist {cnn,mlp} 
                            Model for FID computation on Cifar (default: CNN)
	--batch-size N          Batch size (default: 512)
	--nsamples N            Number of samples per replication (default: 10000)
	--ntests N              Number of replications (default: 3)
	--dataset {cifar10,mnist,celeba} 
                            cifar10, mnist, or celeba 
	--workers WORKERS       Number of data loading workers
	--no-cuda               Disables GPU use
	--sngan                 Enables computing FID for SNGAN

```

### TensorFlow implementation for computing Inception Score and FID

Inception Score: [https://github.com/nnUyi/Inception-Score](https://github.com/nnUyi/Inception-Score)
FID: [author's original implementation](https://github.com/bioinf-jku/TTUR)

## Tested with

- Python 3.6
- Pytorch > 0.4.1
