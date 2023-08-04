import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision.models import resnet18
# from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from utils.data import Dataprocess,Dataprocess_n
from utils.utils import *
from nets.res18 import resnet18_manual
from feature_combine import eu_distance,eu_distance_2
from copy import deepcopy

def test_for_ce():
	arg = create_parser()
	device = "cuda:0"
	num_each_class = arg.num_each_class # 训练权重
	model = resnet18() # 使用在线权重的话服务器登不出去
	# model = mobilenet.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1)
	model.fc = nn.Sequential(
		nn.Linear(512,10),  # 最后的分类个数根据自己数据来 ;resnet18最后是512,mobilenetv2最后是1280
	)
	model.load_state_dict(torch.load(f"logs/mstar_ce_acc_max{num_each_class}.pt"))
	batch_size = arg.batch_size
	model.to(device)
	#data

	tf = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Lambda(lambda x :x.repeat(3,1,1)),
			transforms.Resize((224,224)),
			transforms.Normalize([.5,.5,.5],[.5,.5,.5])
		]
	)
	test_set = Dataprocess_n(data_root = "./data/MSTAR-10",mode = "test",transform=tf,num_each_class = 0) # 0代表全部
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True)
	print(len(test_set))


	save_dir = "logs"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	acc_max = .0
	model.eval()
	pbar_test = tqdm(test_loader)
	correct_all = 0
	all = 0
	j = 0
	with torch.no_grad():
		for x,c in pbar_test:
			x = x.to(device)
			c = c.to(device)
			out = model(x)
			pred = torch.argmax(out,dim = 1)
			correct = (pred==c).sum().cpu().numpy()
			correct_all += correct
			all += c.shape[0]
			pbar_test.set_description(f"correct:{correct}")
			j += 1
	acc = (correct_all / all)
	print(f"acc:{acc : .3f},correct:{correct_all},all:{all}")

def test_for_center():

	arg = create_parser()
	device = "cuda:0"
	num_classes = 10
	num_each_class = arg.num_each_class # 训练权重
	model = resnet18_manual(num_classes = num_classes)
	# model = mobilenet.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1)
	model.fc = nn.Sequential(
		nn.Linear(512,10),  # 最后的分类个数根据自己数据来 ;resnet18最后是512,mobilenetv2最后是1280
	)
	model.load_state_dict(torch.load(f"logs/mstar_center_acc_max{num_each_class}.pt"))
	batch_size = arg.batch_size
	model.to(device)
	#data

	tf = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Lambda(lambda x :x.repeat(3,1,1)),
			transforms.Resize((224,224)),
			transforms.Normalize([.5,.5,.5],[.5,.5,.5])
		]
	)
	test_set = Dataprocess_n(data_root = "./data/MSTAR-10",mode = "test",transform=tf,num_each_class = 0) # 0代表全部
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True)
	print(len(test_set))


	save_dir = "logs"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	acc_max = .0
	model.eval()
	pbar_test = tqdm(test_loader)
	correct_all = 0
	all = 0
	with torch.no_grad():
		for x, c in pbar_test:
			x = x.to(device)
			labels = c.to(device)
			features, outputs = model(x)
			predictions = outputs.data.max(1)[1]
			all += labels.size(0)
			correct_all += (predictions == labels.data).sum()

	acc = (correct_all / all)
	print(f"acc:{acc : .3f},correct:{correct_all},all:{all}")

def test_for_ldlf():


	arg = create_parser()
	device = "cuda:0"
	num_classes = 10
	num_each_class = arg.num_each_class # 训练权重
	model = resnet18_manual(num_classes = num_classes)
	model.fc = nn.Sequential(
		nn.Linear(512,10),  # 最后的分类个数根据自己数据来 ;resnet18最后是512,mobilenetv2最后是1280
	)
	model.load_state_dict(torch.load(f"logs/mstar_ldlf_acc_max{num_each_class}.pt",map_location="cpu"))
	batch_size = arg.batch_size
	model.to(device)
	#data

	tf = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Lambda(lambda x :x.repeat(3,1,1)),
			transforms.Resize((224,224)),
			transforms.Normalize([.5,.5,.5],[.5,.5,.5])
		]
	)
	test_set = Dataprocess_n(data_root = "./data/MSTAR-10",mode = "test",transform=tf,num_each_class = 0) # 0代表全部
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True)
	print(len(test_set))


	save_dir = "logs"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	acc_max = .0
	model.eval()
	pbar_test = tqdm(test_loader)
	correct_all = 0
	all = 0
	with torch.no_grad():
		for x, c in pbar_test:
			x = x.to(device)
			labels = c.to(device)
			features, outputs = model(x)
			predictions = outputs.data.max(1)[1]
			all += labels.size(0)
			correct_all += (predictions == labels.data).sum()

	acc = (correct_all / all)
	print(f"acc:{acc : .3f},correct:{correct_all},all:{all}")
def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("batch_size","--b",type = int ,default = 16)
	arg = parser.parse_args()
	return arg