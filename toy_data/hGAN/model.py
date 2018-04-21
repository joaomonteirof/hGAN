import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Toy data model
class Generator_toy(torch.nn.Module):
	def __init__(self, hidden_dim):
		super(Generator_toy, self).__init__()

		self.all_layers = nn.Sequential(
			nn.Linear(2, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, 2)
			)

	def forward(self, x):
		out = self.all_layers(x)
		return out

class Discriminator_toy(torch.nn.Module):
	def __init__(self, hidden_dim, optimizer, lr, betas):
		super(Discriminator_toy, self).__init__()

		self.projection = nn.utils.weight_norm(nn.Linear(2, 2, bias = False), name = "weight")
		self.projection.weight_g.data.fill_(1)

		self.all_layers = nn.Sequential(
			nn.Linear(2, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(True),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
			)

		self.optimizer = optimizer(list(self.all_layers.parameters()), lr = lr, betas = betas)

	def forward(self, x):
		p_x = self.projection(x)
		out = self.all_layers(p_x)
		return out
