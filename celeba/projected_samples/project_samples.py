import argparse
import glob
import os

import PIL.Image as Image
import torch
import torchvision.transforms as transforms


def denorm(unorm):
	norm = (unorm + 1) / 2
	return norm.clamp(0, 1)


# Training settings
parser = argparse.ArgumentParser(description='Save projected samples')
parser.add_argument('--n-projections', type=int, default=4, metavar='N', help='How many projections per input (default: 4)')
parser.add_argument('--out-path', type=str, default='./out/', metavar='Path', help='Path to output samples')
args = parser.parse_args()

if not os.path.isdir(args.out_path):
	os.mkdir(args.out_path)

input_list = glob.glob('*.jpg')

transform = transforms.Compose([transforms.Resize((64, 64), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

to_pil = transforms.ToPILImage()

for i in range(args.n_projections):

	projection = torch.nn.utils.weight_norm(torch.nn.Conv2d(3, 1, kernel_size=8, stride=2, padding=3, bias=False), name="weight")
	projection.weight_g.data.fill_(1)

	for j, img_path in enumerate(input_list):
		img = Image.open(img_path)
		img = transform(img).unsqueeze(0)
		proj_img = to_pil(denorm(projection(torch.autograd.Variable(img)).data[0]))
		proj_img.save(args.out_path + str(j + 1) + '_' + str(i + 1) + '.png')
