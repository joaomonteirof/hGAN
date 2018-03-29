import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


class ToyData(Dataset):

	def __init__(self, dataset = '8gaussians', length = 100000):

		self.length = length
		self.dataset = dataset	

	def __len__(self):
		
		return self.length

	def __getitem__(self, idx):
			
		if (self.dataset == '8gaussians'):

			scale = 2.
			centers = [
			(1, 0),
			(-1, 0),
			(0, 1),
			(0, -1),
			(1. / np.sqrt(2), 1. / np.sqrt(2)),
			(1. / np.sqrt(2), -1. / np.sqrt(2)),
			(-1. / np.sqrt(2), 1. / np.sqrt(2)),
			(-1. / np.sqrt(2), -1. / np.sqrt(2))
			]

			centers = [(scale * x, scale * y) for x, y in centers]

			sample = np.random.randn(2) * .02
			center = random.choice(centers)
			sample[0] += center[0]
			sample[1] += center[1]

			sample /= 1.414

		if (self.dataset == '25gaussians'):

			range_ = np.arange(-2, 3)
			centers = np.transpose(np.meshgrid(range_, range_, indexing = 'ij'), (1, 2, 0)).reshape(-1, 2)

			center = random.choice(centers)

			sample = np.random.randn(2) * 0.05
			sample[0] += 2 * center[0]
			sample[1] += 2 * center[1]

			sample /= 2.828

				
		sample = {'data': torch.from_numpy(sample).float()}

		return sample
