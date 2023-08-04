from torch.utils.data import Dataset
from PIL import Image
import os
import random

from torchvision.transforms import transforms


class Dataprocess(Dataset):
	def __init__(self,data_root ="",mode = "train",transform = None,data_size = None):
		super().__init__()
		data_dir = ""
		if mode == "train" or mode == "val":
			data_dir = os.path.join(data_root,mode)
		elif mode == "test":
			data_dir = os.path.join(data_root,mode)

		# data
		data_list = []
		classes = []
		for sub_dir in os.listdir(data_dir):
			classes.append(sub_dir)
			sub_dir_path = os.path.join(data_dir,sub_dir)
			target = int(classes.index(sub_dir))
			for file in os.listdir(sub_dir_path):
				file_path = os.path.join(sub_dir_path,file)
				data_list.append((target,file_path))
		self.data_list = data_list
		self.tf = transform
		self.data_size = data_size
		random.seed(1000)
		random.shuffle(self.data_list)
		self.mode = mode
		if self.data_size is not None :
			self.data_list = random.sample(self.data_list,int(len(self.data_list) * self.data_size))
		num_classes = [0 for _ in range(10)]
		for target,file in self.data_list:
			num_classes[target] += 1
		self.num_classes = num_classes
		print(self.num_classes)
	def __len__(self):
		return len(self.data_list)
	def __getitem__(self, item):
		target,data_path = self.data_list[item]
		data = Image.open(data_path)
		if self.tf is not None:
			data = self.tf(data)
		return data,target

class Dataprocess_n(Dataset):
	def __init__(self,data_root ="",mode = "train",transform = None,num_each_class : int = 0):
		super().__init__()
		data_dir = ""
		if mode == "train" or mode == "val":
			data_dir = os.path.join(data_root,mode)
		elif mode == "test":
			data_dir = os.path.join(data_root,mode)

		# data
		data_dic = {}
		data_list = []
		classes = []
		for sub_dir in os.listdir(data_dir):
			if sub_dir not in data_dic:
				data_dic[sub_dir] = []
			classes.append(sub_dir)
			sub_dir_path = os.path.join(data_dir,sub_dir)

			for file in os.listdir(sub_dir_path):
				file_path = os.path.join(sub_dir_path,file)
				data_dic[sub_dir].append(file_path)
		self.data_dic = data_dic
		self.tf = transform
		for target_name,files in self.data_dic.items():
			datas = []
			target = int(classes.index(target_name))
			for file in files:
				datas.append(file)
			# 取数据
			random.seed(1000)
			if num_each_class!=0:
				datas = random.sample(datas,num_each_class)
			for data in datas:
				data_list.append((target,data))
		self.data_list = data_list
	def __len__(self):
		return len(self.data_list)
	def __getitem__(self, item):
		target,data_path = self.data_list[item]
		data = Image.open(data_path)
		if self.tf is not None:
			data = self.tf(data)
		return data,target

if __name__ == "__main__":
	tf = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
			transforms.Resize((224, 224)),
			transforms.Normalize([.5, .5, .5], [.5, .5, .5])
		]
	)
	train_set = Dataprocess_n(data_root="../data/MSTAR-10",mode = "train",transform = tf,num_each_class=10)
	print(len(train_set))