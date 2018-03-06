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

# Training settings
parser = argparse.ArgumentParser(description='Online transfer learning for emotion recognition tasks')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='lambda', help='Adam beta param (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.99, metavar='lambda', help='Adam beta param (default: 0.99)')
parser.add_argument('--ndiscriminators', type=int, default=8, help='Number of discriminators. Default=8')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./celebA', metavar='Path', help='Path to data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--hyper-mode', action='store_true', default=False, help='enables training with hypervolume maximization')
parser.add_argument('--nadir', type=float, default=1.1, metavar='nadir', help='Nadir point for the case of hypervolume maximization (default: 1.1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.RandomGrayscale(0.2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

celebA_data = datasets.ImageFolder(args.data_path, transform = transform)

train_loader = torch.utils.data.DataLoader(celebA_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

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
	trainer = TrainLoop(generator, disc_list, optimizer, train_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, nadir=args.nadir, cuda=args.cuda)
else:
	trainer = TrainLoop(generator, disc_list, optimizer, train_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
