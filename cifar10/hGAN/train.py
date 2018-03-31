from __future__ import print_function

import argparse
import os
import pickle

import PIL.Image as Image
import model
import numpy as np
import resnet
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from train_loop import TrainLoop


def save_testdata_statistics(model, data_loader, cuda_mode):
	for batch in data_loader:

		x, y = batch

		x = torch.autograd.Variable(x)

		out = model.forward(x).data.cpu().numpy()

		try:
			logits = np.concatenate([logits, out], 0)
		except NameError:
			logits = out

	m = logits.mean(0)
	C = np.cov(logits, rowvar=False)

	pfile = open('../test_data_statistics.p', "wb")
	pickle.dump({'m': m, 'C': C}, pfile)
	pfile.close()


# Training settings
parser = argparse.ArgumentParser(description='Hyper volume training of GANs')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='lambda', help='Adam beta param (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
parser.add_argument('--ndiscriminators', type=int, default=8, help='Number of discriminators. Default=8')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='../data', metavar='Path', help='Path to data')
parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--hyper-mode', action='store_true', default=False, help='enables training with hypervolume maximization')
parser.add_argument('--nadir-slack', type=float, default=1.0, metavar='nadir', help='maximum distance to a nadir point component (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.fid_model_path is None:
	print('The path for a pretrained classifier is expected to calculate FID-c. Use --fid-model-path to specify the path')
	exit(1)

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers)

generator = model.Generator(100, [1024, 512, 256, 128], 3).train()
fid_model = resnet.ResNet18().eval()
mod_state = torch.load(args.fid_model_path, map_location=lambda storage, loc: storage)
fid_model.load_state_dict(mod_state['model_state'])

if args.cuda:
	generator = generator.cuda()

if not os.path.isfile('../test_data_statistics.p'):
	testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.workers)
	save_testdata_statistics(fid_model, test_loader, cuda_mode=args.cuda)

disc_list = []

for i in range(args.ndiscriminators):
	disc = model.Discriminator(3, [128, 256, 512, 1024], 1, optim.Adam, args.lr, (args.beta1, args.beta2)).train()
	if args.cuda:
		disc = disc.cuda()
	disc_list.append(disc)

optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

if args.hyper_mode:
	trainer = TrainLoop(generator, fid_model, disc_list, optimizer, train_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, nadir_slack=args.nadir_slack, cuda=args.cuda)
else:
	trainer = TrainLoop(generator, fid_model, disc_list, optimizer, train_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))
print('Hyper Mode is: {}'.format(args.hyper_mode))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
