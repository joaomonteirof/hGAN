from __future__ import print_function

import argparse

import os
import sys
import glob

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 3)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import torch.utils.data

import pandas as pd
import seaborn as sns
import pickle

from common.generators import Generator_mnist
from common.utils import *
from common.models_fid import *
from common.metrics import compute_fid, compute_fid_real_data
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-folder', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--ntests', type=int, default=30, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--nsamples', type=int, default=10000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--fid-model-path', type=str, default=None, metavar='Path', help='Path to fid model')
	parser.add_argument('--data-stat-path', type=str, default='../test_data_statistics.p', metavar='Path', help='Path to file containing test data statistics')
	parser.add_argument('--data-path', type=str, default='../data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./boxplot_data.p', metavar='Path', help='file for dumping boxplot data')
	parser.add_argument('--model-mnist', choices=['cnn', 'mlp'], default='cnn', help='model for FID computation on Cifar. (Default=cnn)')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	
	if args.model_mnist == 'cnn':
		fid_model = cnn().eval()

	elif args.model_mnist == 'mlp':
		fid_model = mlp().eval()

	mod_state = torch.load(args.fid_model_path, map_location = lambda storage, loc: storage)
	fid_model.load_state_dict(mod_state['model_state'])

	if args.cuda:
		fid_model = fid_model.cuda()

	mod_state = torch.load(args.fid_model_path, map_location = lambda storage, loc: storage)
	fid_model.load_state_dict(mod_state['model_state'])


	
	models_dict = {'hyper': 'HV', 'vanilla': 'AVG', 'gman': 'GMAN', 'mgd': 'MGD'}

	fid_dict = {}

	pfile = open(args.data_stat_path, 'rb')
	statistics = pickle.load(pfile)
	pfile.close()

	m, C = statistics['m'], statistics['C']


	if args.cp_folder is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	files_list = glob.glob(args.cp_folder + 'G_*.pt')
	files_list.sort()

	
	for file_id in files_list:

		file_name = file_id.split('/')[-1].split('_')[1]

		print(file_name)		

		key = models_dict[file_name]

		generator = Generator_mnist().eval()
		gen_state = torch.load(file_id, map_location = lambda storage, loc: storage)
		generator.load_state_dict(gen_state['model_state'])

		if args.cuda:
			generator = generator.cuda()
		
		fid = []

		for i in range(args.ntests):
			fid.append(compute_fid(generator, fid_model, args.batch_size, args.nsamples, m, C, args.cuda, inception = False, mnist = True))

		fid_dict[key] = fid


	# Random generator
	random_generator = Generator_mnist().eval()
	
	if args.cuda:
		random_generator = random_generator.cuda()
	
	fid_random = []
	for i in range(args.ntests):
			fid_random.append(compute_fid(random_generator, fid_model, args.batch_size, args.nsamples, m, C, args.cuda, inception = False, mnist = True))
	
	# Real data
	transform = transforms.Compose([transforms.Resize((28, 28), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
	fid_real = compute_fid_real_data(train_loader, fid_model, m, C, args.cuda, inception = False, mnist = True)


	print(np.mean(fid_random))
	print(fid_real)


	df = pd.DataFrame(fid_dict)
	df.head()
	order_plot = ['AVG', 'GMAN', 'HV', 'MGD']
	box = sns.boxplot(data = df, palette = "Set3", width = 0.2, linewidth = 1.0, showfliers = False, order = order_plot)
	box.set_xlabel('Model', fontsize = 12)
	box.set_ylabel('FID - MNIST', fontsize = 12)	
	box.set_yscale('log')
	plt.axhline(np.mean(fid_random), color='r', linestyle='dashed', linewidth=1)
	#plt.axhline(fid_real, color='b', linestyle='dashed', linewidth=1)
	plt.grid(True, alpha = 0.3, linestyle = '--')
	plt.savefig('FID_best_models_mnist.pdf')
	plt.show()

	fid_dict['random']=fid_random
	fid_dict['real']=fid_real

	pfile = open(args.out_file, "wb")
	pickle.dump(fid_dict, pfile)
	pfile.close()
