"""
对于集成学习，
"""
import torch
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from nets.res18 import resnet18_manual
from utils.data import Dataprocess


def random_forest():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# data and model
	model = resnet18_manual(num_classes=10) # 使用在线权重的话服务器登不出去
	batch_size = 16
	model.load_state_dict(torch.load("logs/mstar_ldlf_acc_max.pt",map_location= "cuda:0"))
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

	batch_size = 32
	train_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "train",transform = tf,data_size=0.1)
	train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True,drop_last=False)
	test_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "test",transform = tf)
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True,drop_last=False)
	# 得到Tensor数据
	model.eval()


	correct_all = 0
	all = 0
	j = 0
	train_features_all = []
	train_c_all = []
	with torch.no_grad():
		pbar_train = tqdm(train_loader)
		for x, c in pbar_train:
			batch_ = x.shape[0]
			x = x.to(device)
			c = c.to(device)
			features,out = model(x)
			pred = torch.argmax(out, dim=1)
			correct = (pred == c).sum().cpu().numpy()
			correct_all += correct
			all += c.shape[0]
			pbar_train.set_description(f"correct:{correct}")
			j += 1
			train_features_all.append(features)
			train_c_all.append(c)
	print(f"训练集准确率：{correct_all/all:.4f}")
	print(f"features.shape:{features.shape}")
	train_features_all = torch.concatenate(train_features_all,dim = 0)
	train_c_all = torch.concatenate(train_c_all,dim = 0)
	print(f"train_c_all.shape:{train_c_all.shape},train_features_all.shape:{train_features_all.shape}")

	#将特征及标签转化Tensor转化为numpy
	train_features_all = train_features_all.cpu().numpy()
	train_c_all = train_c_all.cpu().numpy()

	# 构建集成分类器,构建需要用训练数据集，但是预测用
	rfc = RandomForestClassifier(n_estimators = 50)
	rfc.fit(train_features_all,train_c_all) # 构建出分类器

	# 用测试数据进行测试
	pbar_test = tqdm(test_loader)
	correct_all = 0
	all = 0
	j = 0
	test_features_all = []
	test_c_all = []
	with torch.no_grad():
		for x, c in pbar_test:
			batch_ = x.shape[0]
			x = x.to(device)
			c = c.to(device)
			features,out = model(x)
			pred = torch.argmax(out, dim=1)
			correct = (pred == c).sum().cpu().numpy()
			correct_all += correct
			all += c.shape[0]
			pbar_test.set_description(f"correct:{correct}")
			j += 1
			test_features_all.append(features)
			test_c_all.append(c)
	print(f"未经过随机森林前，测试集准确率：{correct_all/all}")
	test_features_all = torch.concatenate(test_features_all,dim = 0)
	test_c_all = torch.concatenate(test_c_all,dim = 0)

	test_features_all = test_features_all.cpu().numpy()
	test_c_all = test_c_all.cpu().numpy()

	test_pred = rfc.predict(test_features_all)
	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(test_c_all,test_pred)
	print(f"经过随机森林后，测试集准确率:{accuracy}")



	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# data and model
	model = resnet18_manual(num_classes=10) # 使用在线权重的话服务器登不出去
	batch_size = 16
	model.load_state_dict(torch.load("logs/mstar_ldlf_acc_max.pt",map_location= "cuda:0"))
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

	batch_size = 32
	train_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "train",transform = tf,data_size=0.1)
	train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True,drop_last=False)
	test_set = Dataprocess(data_root = "./data/MSTAR-10",mode = "test",transform = tf)
	test_loader = DataLoader(test_set,batch_size = batch_size,shuffle = True,drop_last=False)
	# 得到Tensor数据
	model.eval()


	correct_all = 0
	all = 0
	j = 0
	train_features_all = []
	train_c_all = []
	with torch.no_grad():
		pbar_train = tqdm(train_loader)
		for x, c in pbar_train:
			batch_ = x.shape[0]
			x = x.to(device)
			c = c.to(device)
			features,out = model(x)
			pred = torch.argmax(out, dim=1)
			correct = (pred == c).sum().cpu().numpy()
			correct_all += correct
			all += c.shape[0]
			pbar_train.set_description(f"correct:{correct}")
			j += 1
			train_features_all.append(features)
			train_c_all.append(c)
	print(f"训练集准确率：{correct_all/all:.4f}")
	print(f"features.shape:{features.shape}")
	train_features_all = torch.concatenate(train_features_all,dim = 0)
	train_c_all = torch.concatenate(train_c_all,dim = 0)
	print(f"train_c_all.shape:{train_c_all.shape},train_features_all.shape:{train_features_all.shape}")

	#将特征及标签转化Tensor转化为numpy
	train_features_all = train_features_all.cpu().numpy()
	train_c_all = train_c_all.cpu().numpy()

	# 构建集成分类器,构建需要用训练数据集，但是预测用
	adaboost_ = AdaBoostClassifier( n_estimators = 50,base_estimator=rfc)
	adaboost_.fit(train_features_all,train_c_all) # 构建出分类器

	# 用测试数据进行测试
	pbar_test = tqdm(test_loader)
	correct_all = 0
	all = 0
	j = 0
	test_features_all = []
	test_c_all = []
	with torch.no_grad():
		for x, c in pbar_test:
			batch_ = x.shape[0]
			x = x.to(device)
			c = c.to(device)
			features,out = model(x)
			pred = torch.argmax(out, dim=1)
			correct = (pred == c).sum().cpu().numpy()
			correct_all += correct
			all += c.shape[0]
			pbar_test.set_description(f"correct:{correct}")
			j += 1
			test_features_all.append(features)
			test_c_all.append(c)
	print(f"未经过adaboost_前，测试集准确率：{correct_all/all}")
	test_features_all = torch.concatenate(test_features_all,dim = 0)
	test_c_all = torch.concatenate(test_c_all,dim = 0)

	test_features_all = test_features_all.cpu().numpy()
	test_c_all = test_c_all.cpu().numpy()

	test_pred = adaboost_.predict(test_features_all)
	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(test_c_all,test_pred)
	print(f"经过adaboost_后，测试集准确率:{accuracy}")
if __name__ == "__main__":
	random_forest()
	# adaboost()