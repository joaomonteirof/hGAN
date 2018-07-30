import torch.nn as nn
import torch.nn.functional as F

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name='VGG16', soft=False):
		super(VGG, self).__init__()
		self.features = self._make_layers(cfg[vgg_name])
		self.classifier = nn.Linear(512, 10)
		self.soft = soft
		self.downsample = nn.MaxPool2d(2)

	def forward(self, x, downsample_=True):

		if downsample_:
			x = self.downsample(x)

		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		if self.soft:
			out = F.softmax(out, dim=1)
			return F.log_softmax(out, dim=1)
		else:
			return F.log_softmax(out, dim=1)

	def forward_oltl(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return F.log_softmax(out, dim=1), F.softmax(out, dim=1)

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes))

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion * planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes))

	def forward(self, x, downsample_=True):

		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10, soft=False):
		super(ResNet, self).__init__()
		self.in_planes = 64
		self.soft = soft

		self.downsample = nn.MaxPool2d(2)

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, downsample_=True):

		if downsample_:
			x = self.downsample(x)

		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		if self.soft:
			out = F.softmax(out, dim=1)
			return F.log_softmax(out, dim=1)
		else:
			return F.log_softmax(out, dim=1)

	def forward_oltl(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return F.log_softmax(out, dim=1), F.softmax(out, dim=1)


def ResNet18(soft=False):
	return ResNet(BasicBlock, [2, 2, 2, 2], soft=soft)


def ResNet34(soft=False):
	return ResNet(BasicBlock, [3, 4, 6, 3], soft=soft)


def ResNet50(soft=False):
	return ResNet(Bottleneck, [3, 4, 6, 3], soft=soft)


def ResNet101(soft=False):
	return ResNet(Bottleneck, [3, 4, 23, 3], soft=soft)


def ResNet152(soft=False):
	return ResNet(Bottleneck, [3, 8, 36, 3], soft=soft)


class cnn(nn.Module):
	def __init__(self):
		super(cnn, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x, downsample_=False, pre_softmax=False):
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

	def forward_fid(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)

class mlp(nn.Module):
	def __init__(self):
		super(mlp, self).__init__()
		self.fc1 = nn.Linear(784, 320)
		self.fc2 = nn.Linear(320, 50)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x, pre_softmax = False, downsample_=False):
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
