from __future__ import print_function

import argparse

import PIL.Image as Image
import model
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
parser.add_argument('--ndiscriminators', type=int, default=8, help='Number of discriminators. Default=8')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data', metavar='Path', help='Path to data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--hyper-mode', action='store_true', default=False, help='enables training with hypervolume maximization')
parser.add_argument('--nadir-slack', type=float, default=1.0, metavar='nadir', help='maximum distance to a nadir point component (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.LSUN(db_path=args.data_path, classes=['bedroom_train'], transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers)

generator = model.Generator(100, [1024, 512, 256, 128], 3).train()

if args.cuda:
	generator = generator.cuda()

disc_list = []

for i in range(args.ndiscriminators):
	disc = model.Discriminator(3, [128, 256, 512, 1024], 1, optim.Adam, args.lr, (args.beta1, args.beta2)).train()
	if args.cuda:
		disc = disc.cuda()
	disc_list.append(disc)

optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

if args.hyper_mode:
	trainer = TrainLoop(generator, disc_list, optimizer, train_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, nadir_slack=args.nadir_slack, cuda=args.cuda)
else:
	trainer = TrainLoop(generator, disc_list, optimizer, train_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))
print('Hyper Mode is: {}'.format(args.hyper_mode))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
