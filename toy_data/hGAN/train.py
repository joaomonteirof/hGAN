from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import argparse
from common.discriminators import Discriminator_toy
from common.generators import Generator_toy
import torch.optim as optim
import torch.utils.data
from train_loop import TrainLoop
# Training settings
from common.toy_data import ToyData

parser = argparse.ArgumentParser(description='Hyper volume training of GANs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--mgd-lr', type=float, default=0.01, metavar='LR', help='learning rate for mgd (default: 0.01)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='lambda', help='Adam beta param (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
parser.add_argument('--ndiscriminators', type=int, default=8, help='Number of discriminators. Default=8')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=10, metavar='N', help='how many epochs to wait before logging training status. Default is 3')
parser.add_argument('--train-mode', choices=['vanilla', 'hyper', 'gman', 'gman_grad', 'loss_delta', 'mgd'], default='vanilla', help='Salect train mode. Default is vanilla (simple average of Ds losses)')
parser.add_argument('--nadir-slack', type=float, default=1.5, metavar='nadir', help='factor for nadir-point update. Only used in hyper mode (default: 1.5)')
parser.add_argument('--alpha', type=float, default=0.8, metavar='alhpa', help='Used in GMAN and loss_del modes (default: 0.8)')
parser.add_argument('--toy-dataset', choices=['8gaussians', '25gaussians'], default='8gaussians')
parser.add_argument('--toy-length', type=int, metavar='N', help='Toy dataset length', default=100000)
parser.add_argument('--job-id', type=str, default=None, help='Arbitrary id to be written on checkpoints')
parser.add_argument('--sgd', action='store_true', default=False, help='enables SGD - *MGD only* ')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

disc_list = []

toy_data = ToyData(args.toy_dataset, args.toy_length)
train_loader = torch.utils.data.DataLoader(toy_data, batch_size=args.batch_size, num_workers=args.workers)

centers = toy_data.get_centers()
cov = toy_data.get_cov()

# hidden_size = 512
generator = Generator_toy(512).train()

for i in range(args.ndiscriminators):
	disc = Discriminator_toy(512, optim.Adam, args.lr, (args.beta1, args.beta2)).train()
	disc_list.append(disc)

if args.cuda:
	generator = generator.cuda()
	for disc in disc_list:
		disc = disc.cuda()
	torch.backends.cudnn.benchmark=True

if args.train_mode == 'mgd' and args.sgd:
	optimizer = optim.SGD(generator.parameters(), lr=args.mgd_lr)
else:
	optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
trainer = TrainLoop(generator, disc_list, optimizer, args.toy_dataset, centers, cov, train_loader=train_loader, nadir_slack=args.nadir_slack, alpha=args.alpha, train_mode=args.train_mode, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, job_id=args.job_id)

print('Cuda Mode is: {}'.format(args.cuda))
print('Train Mode is: {}'.format(args.train_mode))
print('Number of discriminators is: {}'.format(len(disc_list)))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
