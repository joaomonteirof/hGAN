import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
	def __init__(self):
		super(cnn, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x, pre_softmax=False):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)

		if pre_softmax:
			return x
		else:
			return F.log_softmax(x, dim=1)

	def forward_oltl(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)

		return [F.log_softmax(x, dim=1), F.softmax(x, dim=1)]


class cnn_soft(nn.Module):
	def __init__(self):
		super(cnn_soft, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		y = F.softmax(x, dim=1)

		return F.log_softmax(y, dim=1)


class mlp(nn.Module):
	def __init__(self):
		super(mlp, self).__init__()
		self.fc1 = nn.Linear(784, 320)
		self.fc2 = nn.Linear(320, 50)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x, pre_softmax=False):
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc3(x))

		if pre_softmax:
			return x
		else:
			return F.log_softmax(x, dim=1)

	def forward_oltl(self, x):
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc3(x))

		return [F.log_softmax(x, dim=1), F.softmax(x, dim=1)]


class mlp_soft(nn.Module):
	def __init__(self):
		super(mlp_soft, self).__init__()
		self.fc1 = nn.Linear(784, 320)
		self.fc2 = nn.Linear(320, 50)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x, pre_softmax=False):
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc3(x))

		y = F.softmax(x, dim=1)
		return F.log_softmax(y, dim=1)
