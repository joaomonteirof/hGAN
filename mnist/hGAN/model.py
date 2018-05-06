import torch.nn as nn


class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(100, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 784),
			nn.Tanh()
		)

	def forward(self, x):
		x = x.view(x.size(0), 100)
		out = self.model(x)
		return out


class Discriminator(nn.Module):
	def __init__(self, optimizer, lr, betas):
		super().__init__()

		self.projection = nn.utils.weight_norm(nn.Linear(784, 512), name="weight")
		self.projection.weight_g.data.fill_(1)

		self.hidden_layer = nn.Sequential(
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

		self.optimizer = optimizer(self.hidden_layer.parameters(), lr=lr, betas=betas)

	def forward(self, x):
		p_x = self.projection(x.view(x.size(0), 784))
		out = self.hidden_layer(p_x)
		return out
