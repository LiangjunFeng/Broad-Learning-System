import numpy as np
from sklearn import preprocessing
from numpy import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count/len(Label),5))

class node_generator:
    def __init__(self,whiten = False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten
    
    def sigmoid(self,data):
        return 1.0/(1+np.exp(-data))
    
    def linear(self,data):
        return data
    
    def tanh(self,data):
        return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    
    def relu(self,data):
        return np.maximum(data,0)
    
    def orth(self,W):
        for i in range(0,W.shape[1]):
            w = np.mat(W[:,i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:,j].copy()).T
                w_sum += (w.T.dot(wj))[0,0]*wj 
            w -= w_sum
            w = w/np.sqrt(w.T.dot(w))
            W[:,i] = np.ravel(w)
        return W
        
    def generator(self,shape,times):
        for i in range(times):
            W = 2*random.random(size=shape)-1
            if self.whiten == True:
                W = self.orth(W)
            b = 2*random.random()-1
            yield (W,b)
    
    def generator_nodes(self, data, times, batchsize, nonlinear):
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1],batchsize),times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1],batchsize),times)]
        
        self.nonlinear = {'linear':self.linear,
                          'sigmoid':self.sigmoid,
                          'tanh':self.tanh,
                          'relu':self.relu
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i])+self.blist[i])))
        return nodes
        
    def transform(self,testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i])+self.blist[i])))
        return testnodes   

    def update(self,otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb
        
class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
    
    def fit_transform(self,traindata):
        self._mean = traindata.mean(axis = 0)
        self._std = traindata.std(axis = 0)
        return (traindata-self._mean)/self._std
    
    def transform(self,testdata):
        return (testdata-self._mean)/self._std
        

class broadnet:
    def __init__(self, 
                 maptimes = 10, 
                 enhencetimes = 10,
                 map_function = 'linear',
                 enhence_function = 'linear',
                 batchsize = 'auto', 
                 reg = 0.001):
        
        self._maptimes = maptimes
        self._enhencetimes = enhencetimes
        self._batchsize = batchsize
        self._reg = reg
        self._map_function = map_function
        self._enhence_function = enhence_function
        
        self.W = 0
        self.pesuedoinverse = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse = False)
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten = True)

    def fit(self,data,label):
        if self._batchsize == 'auto':
            self._batchsize = data.shape[1]
        data = self.normalscaler.fit_transform(data)
        label = self.onehotencoder.fit_transform(np.mat(label).T)
        
        mappingdata = self.mapping_generator.generator_nodes(data,self._maptimes,self._batchsize,self._map_function)
        enhencedata = self.enhence_generator.generator_nodes(mappingdata,self._enhencetimes,self._batchsize,self._enhence_function)
        
        print('number of mapping nodes {0}, number of enhence nodes {1}'.format(mappingdata.shape[1],enhencedata.shape[1]))
        print('mapping nodes maxvalue {0} minvalue {1} '.format(round(np.max(mappingdata),5),round(np.min(mappingdata),5)))
        print('enhence nodes maxvalue {0} minvalue {1} '.format(round(np.max(enhencedata),5),round(np.min(enhencedata),5)))
        
        inputdata = np.column_stack((mappingdata,enhencedata))
        pesuedoinverse = self.pinv(inputdata,self._reg)
        self.W =  pesuedoinverse.dot(label)
            
    def pinv(self,A,reg):
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
    
    def decode(self,Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i,:]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)
    
    def accuracy(self,predictlabel,label):
        label = np.ravel(label).tolist()
        predictlabel = predictlabel.tolist()
        count = 0
        for i in range(len(label)):
            if label[i] == predictlabel[i]:
                count += 1
        return (round(count/len(label),5))
        
    def predict(self,testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhencedata = self.enhence_generator.transform(test_mappingdata)
        
        test_inputdata = np.column_stack((test_mappingdata,test_enhencedata))    
        return self.decode(test_inputdata.dot(self.W))      

#data = pd.read_excel('/Users/zhuxiaoxiansheng/Desktop/日常/数据集/糖尿病数据集.xlsx')
#label = np.mat(data['标签'].values)
#data = data.drop('标签',axis = 1)
#traindata,testdata,trainlabel,testlabel = train_test_split(data.values,np.ravel(label),test_size=0.2,random_state = 2018)

#iris=load_iris()
#X = iris.data
#Y = iris.target
#traindata,testdata,trainlabel,testlabel = train_test_split(X,Y,test_size=0.2,random_state = 2018)

data = pd.read_excel('/Users/zhuxiaoxiansheng/Desktop/日常/数据集/wine.xlsx')
label = np.mat(data['Y'].values).T
label -= 1
label = np.ravel(label)
data = data.drop('Y',axis = 1).values
traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2)

bls = broadnet(maptimes = 100, 
               enhencetimes = 100,
               map_function = 'tanh',
               enhence_function = 'relu',
               batchsize = 'auto', 
               reg = 0.001)

bls.fit(traindata,trainlabel)
predictlabel = bls.predict(testdata)
print(show_accuracy(predictlabel,testlabel))












