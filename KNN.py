import numpy as np
import os
import operator
#返回inputdata所属的种类
def KNN(inputdata,TrainingSet,lable,k):
	m=TrainingSet.shape[0] #训练集大小
	difmaze=np.tile(inputdata,(m,1))-TrainingSet #距离矩阵，第i行代表inputdata与第i个训练样例的距离
	sqdifmaze=difmaze ** 2## 距离的平方
	sqsum=sqdifmaze.sum(axis=1) ## 计算每一行的和
	distance=sqsum ** 0.5 ## 欧几里得距离
	sorteddistanceID=distance.argsort() ## 欧几里得距离从小到大排序后的下标
	classcount={} ## 计数器
	for i in range(k): ## 前k近的lable
		nowlable=lable[sorteddistanceID[i]] ##对每个label计数 
		classcount[nowlable]=classcount.get(nowlable,0)+1
	sortedClasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClasscount[0][0] ## 返回出现次数最多的
# 对每个32*32的数字向量化为1*1024的向量
def Vectorfy(filename):
	vec=[]
	fr=open(filename)
	for i in range(32):
		lineStr=fr.readline()
		for j in range(32):
			vec.append(int(lineStr[j]))
	return vec;
def Getlable(filename):
	return filename[0]
# 获取训练集	
def TrainingSet():
	Label=[]
	traininglst=os.listdir('trainingDigits')
	m=len(traininglst)
	trainingmat=np.zeros((m,1024))# 训练矩阵
	for i in range(m):
		filenamestr=traininglst[i]
		Label.append(Getlable(filenamestr))
		trainingmat[i,:]=Vectorfy('trainingDigits/%s' %filenamestr)
	return Label,trainingmat
def Test():#测试测试集
	testlst=os.listdir('testDigits')
	n=len(testlst)
	Lable=[]
	testmat=np.zeros((n,1024))
	for i in range(0,n):
		filenamestr=testlst[i]
		Lable.append(Getlable(filenamestr))
		testmat[i,:]=Vectorfy('testDigits/%s' %filenamestr)
	return Lable,testmat
# if __name__=="main":
testlable,testmat=Test()
trainlabel,trainingmat=TrainingSet()
n=testmat.shape[0]
for k in range(1,20):
	err=0.0
	for i in range(n):
		actlable=KNN(testmat[i],trainingmat,trainlabel,k)
		#print("The correct answer is %d and the actual answer is %d" %(int(testlable[i]),int(actlable)))
		if(testlable[i]!=actlable):
			err+=1
	print('k is {} and the correct rate is {}%'.format(k,(n-err)*100/n))




