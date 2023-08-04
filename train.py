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
from utils.data import Dataprocess
from utils.utils import *
from nets.res18 import resnet18_manual
from feature_combine import eu_distance,eu_distance_2
from copy import deepcopy

# center_loss
class CenterLoss(nn.Module):
	"""Center loss.

	Reference:
	Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

	Args:
		num_classes (int): number of classes.
		feat_dim (int): feature dimension.
	"""

	def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.use_gpu = use_gpu

		if self.use_gpu:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

	def forward(self, x, labels):
		"""
		Args:
			x: feature matrix with shape (batch_size, feat_dim).
			labels: ground truth labels with shape (batch_size).
		"""
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
				  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())

		classes = torch.arange(self.num_classes).long()
		if self.use_gpu: classes = classes.cuda()
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = distmat * mask.float()
		loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

		return loss

def Res_ce():
	# model
	device = "cuda:0"
	model = resnet18() # 使用在线权重的话服务器登不出去
	# model = mobilenet.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1)
	model.fc = nn.Sequential(
		nn.Linear(512,10),  # 最后的分类个数根据自己数据来 ;resnet18最后是512,mobilenetv2最后是1280
	)
	batch_size = 16
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

	train_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "train",transform = tf,data_size=0.1)
	train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
	print(len(train_set))
	test_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "test",transform = tf)
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True)
	print(len(test_set))
	n_epoch = 50
	lr = 0.0003
	op = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)
	lr_scheduler = optim.lr_scheduler.StepLR(op,step_size = 1,gamma = 0.94) # 调整学习率
	criterion = torch.nn.CrossEntropyLoss()
	criterion = criterion.to(device)

	save_dir = "logs"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	acc_max = .0
	for ep in range(n_epoch):
		pbar = tqdm(train_loader)
		model.train()
		j = 0
		for x,c in pbar:
			x = x.to(device)
			c = c.to(device)
			out = model(x)
			# pred = torch.argmax(out,dim = 1)
			loss = criterion(out,c)
			op.zero_grad()
			loss.backward()
			op.step()
			pbar.set_description(f"epoch:{ep},loss:{loss.item():.4f}")
			j += 1
		model.eval()
		pbar_test = tqdm(test_loader)
		correct_all = 0
		all = 0
		j = 0
		with torch.no_grad():
			for x,c in pbar_test:
				batch_ = x.shape[0]
				x = x.to(device)
				c = c.to(device)
				out = model(x)
				pred = torch.argmax(out,dim = 1)
				correct = (pred==c).sum().cpu().numpy()
				correct_all += correct
				all += c.shape[0]
				pbar_test.set_description(f"epoch:{ep},correct:{correct}")
				j += 1
		print(f"ep:{ep},acc:{correct_all/all : .3f},correct:{correct_all},all:{all}")
		acc = (correct_all/all)
		lr_scheduler.step()
		if acc >acc_max:
			acc_max = acc
			torch.save(model.state_dict(),"logs/mstar_ce_acc_max.pt")
		print(f"max_acc:{acc_max}")

def Res_LDLF():
	# model
	device = "cuda:0"
	# device = "cpu"
	model = resnet18_manual(num_classes=10)
	# 有关预训练权重的使用，先存疑
	batch_size = 16
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

	train_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "train",transform = tf,data_size=0.1)
	train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
	print(len(train_set))
	test_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "test",transform = tf)
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True)
	print(len(test_set))
	n_epoch = 50
	lr = 0.0003
	op = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)
	lr_scheduler = optim.lr_scheduler.StepLR(op,step_size = 1,gamma = 0.94) # 调整学习率
	criterion_ce = torch.nn.CrossEntropyLoss().to(device)
	# 超参 margin,lambda
	margin = .5
	lambda_ = 1. #lambda 是python关键字
	# 对比损失的第二部分的书写 就是欧式距离等，不需要nn里面集成的损失函数
	acc_min = .0
	for ep in range(n_epoch):
		pbar = tqdm(train_loader)
		model.train()
		j = 0
		loss2_ema = None
		for x,c in pbar: # 还存在一个问题是，输出的features是Tensor，不用转成list，直接Tensor计算就可以。
			x = x.to(device)
			c = c.to(device)
			features,out = model(x)
			# pred = torch.argmax(out,dim = 1)
			loss_ce = criterion_ce(out,c)
			batch_ = x.shape[0]
			labels = c

			# 开始计算对比损失
			# 是否同一类别的标签
			# y_ijs = [[0] * batch_ for _ in range(batch_)]
			# for i in range(batch_):
			# 	for j in range(batch_):
			# 		y_ijs[i][j] = 1 if labels[i] == labels[j] else 0
			loss2 = torch.Tensor([.0]).to(device)
			# print(f"features.shape:{features.shape}")
			for i in range(1,batch_):
				features1 = features.clone()
				labels1 = labels.clone()
				# 得到features1和lobels1
				for k in range(0,batch_):
					features1[k] = features[(k+i)%batch_]
					labels1[k] = labels[(k+i)%batch_]
				y_ij = (labels1.eq(labels))
				# print(f"y_ij:{y_ij}")
				# 是否同一标签的记录y
				# 开始计算损失
				d_matrix = eu_distance(features1,features)
				# print(d_matrix)
				for i in range(batch_):
					d_2 = d_matrix[i][i] ** 2 #平方
					# print(f"d_2:{d_2},type(d_2):{type(d_2)}")

					loss2 += int(y_ij[i]) * d_2 + int(1-int(y_ij[i])) * max(margin-d_matrix[i][i],0) ** 2
			loss2 = loss2.item() / (2 * batch_ * (batch_ -1) )
			# print(f"loss2:{loss2},type(loss2):{type(loss2)}")

			loss = loss_ce + lambda_ * loss2
			op.zero_grad()
			loss.backward()
			op.step()
			pbar.set_description(f"epoch:{ep},loss:{loss.item():.4f}")
			j += 1
		model.eval()
		pbar_test = tqdm(test_loader)
		correct_all = 0
		all = 0
		j = 0
		with torch.no_grad():
			for x,c in pbar_test:
				batch_ = x.shape[0]
				x = x.to(device)
				c = c.to(device)
				features,out = model(x)
				pred = torch.argmax(out,dim = 1)
				correct = (pred==c).sum().cpu().numpy()
				correct_all += correct
				all += c.shape[0]
				pbar_test.set_description(f"epoch:{ep},correct:{correct}")
				j += 1
		print(f"ep:{ep},acc:{correct_all/all : .3f},correct:{correct_all},all:{all}")
		lr_scheduler.step()
		acc = (correct_all/all)
		if acc >acc_min:
			acc_min = acc
			torch.save(model.state_dict(),"logs/mstar_ldlf_acc_max.pt")

def Res_Centerloss():

	# model
	device = "cuda:0"
	# device = "cpu"
	num_classes = 10
	model = resnet18_manual(num_classes = num_classes)
	batch_size = 4
	model.to(device)
	use_gpu = True
	#data
	tf = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Lambda(lambda x :x.repeat(3,1,1)),
			transforms.Resize((224,224)),
			transforms.Normalize([.5,.5,.5],[.5,.5,.5])
		]
	)

	train_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "train",transform = tf,data_size=0.1)
	train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
	print(len(train_set))
	test_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "test",transform = tf)
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True)
	print(len(test_set))
	n_epoch = 50
	lr_model = 0.0003
	lr_cent = 0.5
	weight_cent = 1.
	step_size = 20
	gamma = .5
	# loss and optimizer
	criterion_xent = nn.CrossEntropyLoss()
	# feat_dim 要根据自己输出的特征维度去判断
	criterion_cent = CenterLoss(num_classes = num_classes, feat_dim = 512, use_gpu=use_gpu)
	optimizer_model = torch.optim.SGD(model.parameters(), lr = lr_model, weight_decay=5e-04, momentum=0.9)
	optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr = lr_cent)
	scheduler = lr_scheduler.StepLR(optimizer_model, step_size = step_size, gamma = gamma)

	# 方便记录,AverageMeter
	xent_losses = AverageMeter()
	cent_losses = AverageMeter()
	losses = AverageMeter()
	save_dir = "logs"
	acc_min = .0
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for ep in range(n_epoch):
		pbar = tqdm(train_loader)
		model.train()
		j = 0
		loss2_ema = None
		model.train()
		for x,c in pbar: # 还存在一个问题是，输出的features是Tensor，不用转成list，直接Tensor计算就可以。
			x = x.to(device)
			labels = c.to(device)
			features,outputs = model(x)
			# pred = torch.argmax(out,dim = 1)
			loss_xent = criterion_xent(outputs, labels)
			loss_cent = criterion_cent(features, labels)
			loss_cent *= weight_cent
			loss = loss_xent + loss_cent
			optimizer_model.zero_grad()
			optimizer_centloss.zero_grad()
			loss.backward()
			optimizer_model.step()
			# by doing so, weight_cent would not impact on the learning of centers
			for param in criterion_cent.parameters():
				param.grad.data *= (1. / weight_cent)
			optimizer_centloss.step()

			losses.update(loss.item(), labels.size(0))
			xent_losses.update(loss_xent.item(), labels.size(0))
			cent_losses.update(loss_cent.item(), labels.size(0))
			pbar.set_description(f"epoch:{ep},loss:{loss.item():.4f}")

		model.eval()
		pbar_test = tqdm(test_loader)
		correct, total = 0, 0
		plot = True
		if plot:
			all_features, all_labels = [], []
		with torch.no_grad():
			for x,c in pbar_test:
				batch_ = x.shape[0]
				x = x.to(device)
				labels = c.to(device)
				features, outputs = model(x)
				predictions = outputs.data.max(1)[1]
				total += labels.size(0)
				correct += (predictions == labels.data).sum()

				if plot:
					if use_gpu:
						all_features.append(features.data.cpu().numpy())
						all_labels.append(labels.data.cpu().numpy())
					else:
						all_features.append(features.data.numpy())
						all_labels.append(labels.data.numpy())
			print(f"ep:{ep},acc:{correct / total : .3f},correct:{correct},total:{total}")
			acc = correct * 100. / total
			err = 100. - acc
			# return acc, er
			if acc >acc_min:
				acc_min = acc
				torch.save(model.state_dict(),"logs/mstar_ce_acc_max.pt")


if __name__ == "__main__":
	Res_ce()
	print(f"res_ce done!")
	#
	# Res_Centerloss()
	# print(f"res_centerloss done!")
	# Res_LDLF()
	# print(f"res_ldlf done!")