from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

from common.models_fid import ResNet18
from common.utils import save_testdata_statistics
from common.generators import Generator_SN
from common.discriminators import Discriminator_SN_noproj
import argparse
import os
import PIL.Image as Image
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from train_loop import TrainLoop


# Training settings
parser = argparse.ArgumentParser(description='Hyper volume training of GANs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='lambda', help='Adam beta param (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
parser.add_argument('--its-disc', type=int, default=5, metavar='N', help='D train iterations per G iteration (Default: 5)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='../data', metavar='Path', help='Path to data')
parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.fid_model_path is None:
	print('The path for a pretrained classifier is expected to calculate FID-c. Use --fid-model-path to specify the path')
	exit(1)

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([transforms.Resize((32, 32), interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers)

generator = Generator_SN().train()
disc = Discriminator_SN_noproj(optim.Adam, args.lr, (args.beta1, args.beta2)).train()
fid_model = ResNet18().eval()
mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
fid_model.load_state_dict(mod_state['model_state'])

if args.cuda:
	generator = generator.cuda()
	disc = disc.cuda()
	torch.backends.cudnn.benchmark=True

if not os.path.isfile('../test_data_statistics.p'):
	testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.workers)
	save_testdata_statistics(fid_model, test_loader, downsample_=False, cuda_mode=args.cuda)

optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

trainer = TrainLoop(generator, fid_model, disc, optimizer, train_loader, its_disc=args.its_disc, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
