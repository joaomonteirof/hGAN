import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import argparse
import os
import sys
import h5py

sys.path.insert(0, os.path.realpath(__file__ + ('/..' * 2)))
print(f'Running from package root directory {sys.path[0]}')

def prepare_data(data_path):

	global images

	transform = transforms.Compose([transforms.Resize((28, 28), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	images = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

def get_random_digit_image():
	indx = randint(0, len(images)-1)
	img = images[indx][0]
	return img


def create_dataset(samples):
	dataset = None

	for i in range(samples):
		print(i)
		img_ch0 = get_random_digit_image()
		img_ch1 = get_random_digit_image()
		img_ch2 = get_random_digit_image()

		im = torch.cat([img_ch0, img_ch1, img_ch2], 0).unsqueeze(0)

		if dataset is not None:
			dataset = torch.cat([im, dataset], 0)
		else:
			dataset = im
	return dataset

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create stacked mnist')
	parser.add_argument('--data-size', type=int, default=50000, metavar='N', help='Dataset size (default: 50000)')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./smnist.hdf', metavar='Path', help='output path')
	args = parser.parse_args()

	prepare_data(args.data_path)
	dataset = create_dataset(args.data_size)
	print(dataset.size())

	hdf = h5py.File(args.out_file, 'a')
	hdf.create_dataset('data', data=dataset)
