from __future__ import print_function

import argparse

import os
import sys
import glob

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

import matplotlib.pyplot as plt
import torch.utils.data

import pandas as pd
import seaborn as sns
import pickle

from common.generators import Generator
from common.utils import *
from common.models_fid import *
from common.metrics import compute_diversity_mssim, get_gen_samples
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-folder', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--ntests', type=int, default=5, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--nsamples', type=int, default=10000, metavar='Path', help='number of samples per replication')
	parser.add_argument('--data-path', type=str, default='../data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./celeba_boxplot_data.p', metavar='Path', help='file for dumping boxplot data')
	parser.add_argument('--batch-size', type=int, default=512, metavar='Path', help='batch size')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--no-calc-real', action='store_true', default=False, help='Enables calculating metric for real data')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False


	models_dict = {'hyper8': 'HV-8', 'hyper16': 'HV-16', 'hyper24': 'HV-24', 'vanilla8': 'AVG-8', 'vanilla16': 'AVG-16', 'vanilla24': 'AVG-24', 'gman8': 'GMAN-8', 'gman16': 'GMAN-16', 'gman24': 'GMAN-24'}

	mssim_dict = {}

	if (args.no_calc_real):

		pfile = open('mssim_real_celeba.p', 'rb')
		mean_list = pickle.load(pfile)
		pfile.close()
		real_mean = np.mean(mean_list)

	else:

		transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		celebA_data = datasets.ImageFolder(args.data_path, transform=transform)
		train_loader = iter(torch.utils.data.DataLoader(celebA_data, batch_size=args.nsamples, shuffle=True))
		
		mssim_real = []
		for i in range(args.ntests):
			samples = next(train_loader)[0]
			mssim_real.append(compute_diversity_mssim(samples, real = True, mnist = False))

		real_mean = np.mean(mssim_real)
		pfile = open('mssim_real_celeba.p', "wb")
		pickle.dump(mssim_real, pfile)
		pfile.close()

	print(real_mean)

	if args.cp_folder is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	files_list = glob.glob(args.cp_folder + 'G_*.pt')
	files_list.sort()

	for file_id in files_list:

		file_name = file_id.split('/')[-1].split('_')[1]

		print(file_name)		

		key = models_dict[file_name]

		generator = Generator(100, [1024, 512, 256, 128], 3).eval()
		gen_state = torch.load(file_id, map_location = lambda storage, loc: storage)
		generator.load_state_dict(gen_state['model_state'])

		if args.cuda:
			generator = generator.cuda()
		
		mssim = []

		for i in range(args.ntests):
			samples = get_gen_samples(generator, batch_size=args.batch_size, nsamples=args.nsamples, cuda=args.cuda, mnist=False)
			mssim.append(compute_diversity_mssim(samples, real=False, mnist=False) - real_mean)

		mssim_dict[key] = mssim
		print(mssim)		

	df = pd.DataFrame(mssim_dict)
	df.head()
	order_plot = ['AVG-8', 'GMAN-8', 'HV-8', 'AVG-16', 'GMAN-16', 'HV-16', 'AVG-24', 'GMAN-24', 'HV-24']
	box = sns.boxplot(data = df, palette = "Set3", width = 0.2, linewidth = 1.0, showfliers = False, order = order_plot)
	box.set_xlabel('Model', fontsize = 15)
	box.set_ylabel('Diversity - CelebA', fontsize = 15)	
	#box.set_yscale('log')
	#plt.axhline(real_mean, color='r', linestyle='dashed', linewidth=1)
	plt.savefig('mssim_celeba.pdf')
	plt.show()
	
	pfile = open(args.out_file, "wb")
	pickle.dump(mssim_dict, pfile)
	pfile.close()
