from __future__ import print_function

import argparse
import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

import PIL.Image as Image
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from common.discriminators import *
from common.utils import save_testdata_statistics
from common.generators import Generator
from common.models_fid import cnn

from common.generators import Generator_stacked_mnist
from common.discriminators import Discriminator_stacked_mnist
from train_loop import TrainLoop
from data_load import Loader

parser = argparse.ArgumentParser(description='Hyper volume training of GANs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--mgd-lr', type=float, default=0.01, metavar='LR', help='learning rate for mgd (default: 0.01)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='lambda', help='Adam beta param (default: 0.5), or alpha param for RMSprop')
parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
parser.add_argument('--ndiscriminators', type=int, default=8, help='Number of discriminators. Default=8')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--classifier-path', type=str, default=None, metavar='Path', help='Path to pretrained classifier on MNIST')
parser.add_argument('--data-path', type=str, default='./train.hdf', metavar='Path', help='Path to hdf file containing stacked MNIST. Can be generated with gen_data.py')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--train-mode', choices=['vanilla', 'hyper', 'gman', 'gman_grad', 'loss_delta', 'mgd'], default='vanilla', help='Salect train mode. Default is vanilla (simple average of Ds losses)')
parser.add_argument('--nadir-slack', type=float, default=1.5, metavar='nadir', help='factor for nadir-point update. Only used in hyper mode (default: 1.5)')
parser.add_argument('--alpha', type=float, default=0.8, metavar='alhpa', help='Used in GMAN and loss_del modes (default: 0.8)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--sgd', action='store_true', default=False, help='enables SGD - *MGD only* ')
parser.add_argument('--job-id', type=str, default=None, help='Arbitrary id to be written on checkpoints')
parser.add_argument('--optimizer', choices=['adam', 'amsgrad', 'rmsprop'], default='adam', help='Select optimizer (Default is adam).')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

trainset = Loader(args.data_path)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers)

generator = Generator_stacked_mnist().train()
classifier = cnn().eval()
classifier_state = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
classifier.load_state_dict(classifier_state['model_state'])

disc_list = []

for i in range(args.ndiscriminators):
	if args.optimizer == 'adam':
		disc = Discriminator_stacked_mnist(optim.Adam, args.optimizer, args.lr, (args.beta1, args.beta2)).train()
	elif args.optimizer == 'amsgrad':	
		disc = Discriminator_stacked_mnist(optim.Adam, args.optimizer, args.lr, (args.beta1, args.beta2), amsgrad = True).train()
	elif args.optimizer == 'rmsprop':
		disc = Discriminator_stacked_mnist(optim.RMSprop, args.optimizer, args.lr, (args.beta1, args.beta2)).train()
	disc_list.append(disc)


if args.cuda:
	generator = generator.cuda()
	classifier = classifier.cuda()
	for disc in disc_list:
		disc = disc.cuda()
	torch.backends.cudnn.benchmark=True

if args.train_mode == 'mgd' and args.sgd:
	optimizer_g = optim.SGD(generator.parameters(), lr=args.mgd_lr)
elif args.optimizer == 'adam':
	optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
elif args.optimizer == 'amsgrad':
	optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), amsgrad = True)
elif args.optimizer == 'rmsprop':
	optimizer_g = optim.RMSprop(generator.parameters(), lr=args.lr, alpha = args.beta1)

trainer = TrainLoop(generator, disc_list, optimizer_g, train_loader, classifier=classifier, nadir_slack=args.nadir_slack, alpha=args.alpha, train_mode=args.train_mode, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, job_id=args.job_id)

print('Cuda Mode is: {}'.format(args.cuda))
print('Train Mode is: {}'.format(args.train_mode))
print('Number of discriminators is: {}'.format(len(disc_list)))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
