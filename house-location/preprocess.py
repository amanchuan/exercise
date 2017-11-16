import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.externals import joblib


def LoadData(file):
	data = []
	i = 0
	with open(file, encoding='utf-8') as csv_file:
		csv_reader = csv.reader(csv_file)	
		for row in csv_reader:
			data.append(row)
			i += 1
			if i % 5000 == 0:				
				print(".", end="")
	data = np.array(data)
	return data[1:]

def LoadData2(file):
	data = []
	iterm = 0
	with open(file, encoding='utf-8') as csv_file:
		csv_reader = csv.reader(csv_file)	
		for row in csv_reader:
			data.append(row)
			iterm += 1
			if iterm == 100001: break
			# if iterm % 1000 == 0:
			# 	print(".", end="")
	data = np.array(data)
	return data[1:]

def PreProccess():
	"""
	由于取所有训练数据训练模型，然后预测所有数据有memoryerror。
	所以只取所有训练数据来训练模型，然后保存模型，之后直接用模型预测。
	"""
	shop_inf = pd.read_csv('train-ccf_first_round_shop_info.csv', usecols=[0,2,3,5])
	train_data = pd.read_csv('train-ccf_first_round_user_shop_behavior.csv', usecols=[1])
	shop_inf = np.array(shop_inf)
	train_data = np.array(train_data)
	#test_data = pd.read_csv('ABtest-evaluation_public.csv')
	#print(shop_inf.shape)		# (8477, 6)
	#print(train_data.shape)	# (1048575, 6)
	#print(test_data.shape)		# (483931, 7)
	shop_position = []	
	shop_int_number = []	
	num_train = train_data.shape[0]
	num_shop = shop_inf.shape[0]
	#num_test = test_data.shape[0]
	shop_seq = np.arange(num_shop)		# (8477,)
	for i in range(num_train):
		shop_id = train_data[i]
		for j in range(num_shop):
			if shop_inf[j,0] == shop_id:
				shop_position.append(shop_inf[j,[1,2,3]])		# 在训练数据中追加商店所在的商场id
				#shop_int_number.append(shop_seq[j])		# 追加商店序号id
	shop_position = np.array(shop_position)
	print(shop_position.shape)
	print(shop_position[0,10])
	dataframe = pd.DataFrame(shop_position)
	dataframe.to_csv("train_data_shop_position.csv",index=False)
	# shop_int_number = np.array(shop_int_number).reshape(num_train, -1)
	# new_train_data = np.hstack((train_data, shop_mall_id))		# (10000, 7)
	# new_train_data = np.hstack((new_train_data, shop_int_number))	# (10000, 8)
PreProccess()

def predict_sub():
	#classifier = joblib.load("LR.model")
	test_data = LoadData2('ABtest-evaluation_public.csv')
	num_test = test_data.shape[0]
	new_test_data = test_data[:, [4,5]].astype(np.float64)

	predict = classifier.predict(new_test_data).astype(np.int32)
	predict_shop_id = shop_inf[predict, 0].reshape(num_test, -1)
	test_id = test_data[:,0].reshape(num_test, -1)
	result = np.hstack((test_id, predict_shop_id))
	print(result.shape)

	dataframe = pd.DataFrame({'row_id':test_id, 'shop_id':predict_shop_id})
	dataframe.to_csv("result1.csv")

#PreProccess()
def ExtractData():
	train_data = pd.read_csv('train_data_extracted.csv')
	shop_inf = pd.read_csv('shop_inf_extracted.csv')
	train_data = np.array(train_data)
	shop_inf = np.array(shop_inf)
	num_train = train_data.shape[0]
	num_shop = shop_inf.shape[0]
	shop_seq = np.arange(num_shop)
	shop_id_number = []
	for i in range(num_train):
		shop_id = train_data[i,0]
		shop_id_number.append(np.squeeze(np.argwhere(shop_inf==shop_id)))
		if i % 1000 ==0:
			print(".",end="")
	shop_id_number = np.array(shop_id_number)
	dataframe = pd.DataFrame(shop_id_number)
	dataframe.to_csv("y_train.csv", index=False)

def fit():
	classifier = joblib.load("LR_0.3.model")
	x_train = pd.read_csv('train_data_extracted.csv', usecols=[1,2],skiprows=300000,nrows=100000)
	y_train = pd.read_csv('y_train.csv',usecols=[0], skiprows=300000,nrows=100000)
	x_train = np.array(x_train)
	y_train = np.squeeze(np.array(y_train))

	#classifier = LogisticRegression()

	classifier.fit(x_train,y_train)
	joblib.dump(classifier, "LR_0.4.model")

def predict():
	classifier = joblib.load("LR_0.4.model")
	test_data = pd.read_csv('ABtest-evaluation_public.csv', usecols=[4,5], skiprows=400000 )
	test_id = pd.read_csv('ABtest-evaluation_public.csv', usecols=[0], skiprows=400000 )
	test_data = np.array(test_data)
	test_id = np.squeeze(np.array(test_id))
	shop_id_number = classifier.predict(test_data)

	shop_id = pd.read_csv('shop_inf_extracted.csv')
	shop_id = np.squeeze(np.array(shop_id))
	shop_id_predict = shop_id[shop_id_number]

	dataframe = pd.DataFrame({'row_id':test_id, 'shop_id':shop_id_predict})
	dataframe.to_csv("result_0.4.4.csv", index=False)
	# print("filished")
#fit()
#predict()