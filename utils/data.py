from torch.utils.data import Dataset
from PIL import Image
import os
import random

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
