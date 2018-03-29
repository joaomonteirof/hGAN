from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from train_loop import TrainLoop
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data
import model
import numpy as np

from ToyData import ToyData

import os
import pickle

def save_data_statistics(data_loader, data_statistics_name):

	for batch in data_loader:

		x = batch['data'].cpu().numpy()

		try:
			samples = np.concatenate([samples, x], 0)
		except NameError:
			samples = x

	m = samples.mean(0)
	C = np.cov(samples, rowvar = False)	

	pfile = open(data_statistics_name,"wb")
	pickle.dump({'m': m, 'C': C}, pfile)
	pfile.close()

# Training settings
parser = argparse.ArgumentParser(description='Hyper volume training of GANs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='lambda', help='Adam beta param (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
parser.add_argument('--lambda-grad', type=float, default=10.0, metavar='Lambda', help='lambda for gradient penalty (default: 10.0)')
parser.add_argument('--its-disc', type=int, default=5, metavar='N', help='D train iterations per G iteration (Default: 5)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 3')
parser.add_argument('--toy-dataset', choices=['8gaussians', '25gaussians'], default='8gaussians')
parser.add_argument('--toy-length', type=int, metavar = 'N', help='Toy dataset length', default=100000)
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)


toy_data = ToyData(args.toy_dataset, args.toy_length)
train_loader = torch.utils.data.DataLoader(toy_data, batch_size = args.batch_size, num_workers = args.workers)

data_statistics_name = '../data_statistics' + args.toy_dataset + '.p' 
if not os.path.isfile(data_statistics_name):
	save_data_statistics(train_loader, data_statistics_name)

# hidden_size = 512
generator = model.Generator_toy(512).train()


disc = model.Discriminator_toy(512, optim.Adam, args.lr, (args.beta1, args.beta2)).train()

optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

trainer = TrainLoop(generator, disc, optimizer, data_statistics_name, train_loader = train_loader, lambda_grad=args.lambda_grad, its_disc=args.its_disc, checkpoint_path = args.checkpoint_path, checkpoint_epoch = args.checkpoint_epoch, cuda = args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
