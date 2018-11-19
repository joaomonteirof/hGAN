import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import ResnetBlock

class Generator(torch.nn.Module):
	def __init__(self, input_dim, num_filters, output_dim):
		super(Generator, self).__init__()

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Deconvolutional layer
			if i == 0:
				deconv = nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
			else:
				deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			deconv_name = 'deconv' + str(i + 1)
			self.hidden_layer.add_module(deconv_name, deconv)

			# Initializer
			nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
			nn.init.constant_(deconv.bias, 0.0)

			# Batch normalization
			bn_name = 'bn' + str(i + 1)
			self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.ReLU())

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Deconvolutional layer
		out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', torch.nn.Tanh())

	def forward(self, x):
		h = self.hidden_layer(x)
		out = self.output_layer(h)
		return out

class Generator_SN(nn.Module):
	def __init__(self):
		super().__init__()
		self.z_dim = 128
		self.m_g = 4
		self.ch = 512

		self.linear = nn.Linear(self.z_dim, self.m_g*self.m_g*self.ch)
		self.activation = nn.ReLU()
		self.deconv = nn.Sequential(

			nn.ConvTranspose2d(self.ch, self.ch//2, 4, 2, 1),
			nn.BatchNorm2d(self.ch//2),
			nn.ReLU(),

			nn.ConvTranspose2d(self.ch//2, self.ch//4, 4, 2, 1),
			nn.BatchNorm2d(self.ch//4),
			nn.ReLU(),

			nn.ConvTranspose2d(self.ch//4, self.ch//8, 4, 2, 1),
			nn.BatchNorm2d(self.ch//8),
			nn.ReLU(),

			nn.ConvTranspose2d(self.ch//8, 3, 3, 1, 1),
			nn.Tanh()
		)

	def forward(self, z):
		out = self.activation(self.linear(z))
		out = out.view(-1, self.ch, self.m_g, self.m_g)
		out = self.deconv(out)

		return out

class Generator_stacked_mnist(torch.nn.Module):
	def __init__(self):
		super(Generator_stacked_mnist, self).__init__()

		#linear layer
		self.linear = torch.nn.Sequential()

		linear = nn.Linear(100, 2*2*512)

		self.linear.add_module('linear', linear)

		# Initializer
		nn.init.normal_(linear.weight, mean=0.0, std=0.02)
		nn.init.constant_(linear.bias, 0.0)

		# Batch normalization
		bn_name = 'bn0'
		self.linear.add_module(bn_name, torch.nn.BatchNorm1d(2*2*512))

		# Activation
		act_name = 'act0'
		self.linear.add_module(act_name, torch.nn.ReLU())

		# Hidden layers
		num_filters = [256, 128, 64]
		self.hidden_layer = torch.nn.Sequential()
		for i in range(3):
			# Deconvolutional layer
			if i == 0:
				deconv = nn.ConvTranspose2d(512, num_filters[i], kernel_size=4, stride=2, padding=1)
			elif i == 2:
				deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=2)
			else:
				deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			deconv_name = 'deconv' + str(i + 1)
			self.hidden_layer.add_module(deconv_name, deconv)

			# Initializer
			nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
			nn.init.constant_(deconv.bias, 0.0)

			# Batch normalization
			bn_name = 'bn' + str(i + 1)
			self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.ReLU())

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Deconvolutional layer
		out = torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', torch.nn.Tanh())

	def forward(self, x):

		x = x.view(x.size(0), -1)
		x = self.linear(x)

		h = self.hidden_layer(x.view(x.size(0), 512, 2, 2))
		out = self.output_layer(h)
		return out

class Generator_res(nn.Module):
	def __init__(self, z_dim=128, size=256, nfilter=64):
		super().__init__()
		s0 = self.s0 = size // 32
		nf = self.nf = nfilter
		self.z_dim = z_dim

		# Submodules
		self.fc = nn.Linear(z_dim, 16*nf*s0*s0)

		self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
		self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)

		self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
		self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)

		self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
		self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)

		self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
		self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)

		self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
		self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)

		self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
		self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

		self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

	def forward(self, z):

		batch_size = z.size(0)

		out = self.fc(z)
		out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

		out = self.resnet_0_0(out)
		out = self.resnet_0_1(out)

		out = F.interpolate(out, scale_factor=2)
		out = self.resnet_1_0(out)
		out = self.resnet_1_1(out)

		out = F.interpolate(out, scale_factor=2)
		out = self.resnet_2_0(out)
		out = self.resnet_2_1(out)

		out = F.interpolate(out, scale_factor=2)
		out = self.resnet_3_0(out)
		out = self.resnet_3_1(out)

		out = F.interpolate(out, scale_factor=2)
		out = self.resnet_4_0(out)
		out = self.resnet_4_1(out)

		out = F.interpolate(out, scale_factor=2)
		out = self.resnet_5_0(out)
		out = self.resnet_5_1(out)

		out = self.conv_img(F.leaky_relu(out, 2e-1))
		out = torch.tanh(out)

		return out

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

class Generator_mnist(nn.Module):
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
