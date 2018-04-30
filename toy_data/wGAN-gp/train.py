from __future__ import print_function

import argparse

import model
import torch.optim as optim
import torch.utils.data
from train_loop import TrainLoop

# Training settings
from common.toy_data import ToyData

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
parser.add_argument('--toy-length', type=int, metavar='N', help='Toy dataset length', default=100000)
args = parser.parse_args()

torch.manual_seed(args.seed)

toy_data = ToyData(args.toy_dataset, args.toy_length)
train_loader = torch.utils.data.DataLoader(toy_data, batch_size=args.batch_size, num_workers=args.workers)

centers = toy_data.get_centers()
cov = toy_data.get_cov()

# hidden_size = 512
generator = model.Generator_toy(512).train()

disc = model.Discriminator_toy(512, optim.Adam, args.lr, (args.beta1, args.beta2)).train()

optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

trainer = TrainLoop(generator, disc, optimizer, args.toy_dataset, centers, cov, train_loader=train_loader, lambda_grad=args.lambda_grad, its_disc=args.its_disc, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch)

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
