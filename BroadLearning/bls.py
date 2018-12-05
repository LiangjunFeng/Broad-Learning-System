import numpy as np
from sklearn import preprocessing
from numpy import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import skimage.io as io
import skimage
import math
from sklearn.decomposition import PCA
import scipy.io as sio

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
        return (traindata-self._mean)/(self._std+0.001)
    
    def transform(self,testdata):
        return (testdata-self._mean)/(self._std+0.001)
        

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


def LoadData(number):
    if number == 1:
        path = '/Users/zhuxiaoxiansheng/Desktop/日常/数据集/yale_faces/*.bmp'
    elif number == 2:
        path = '/Users/zhuxiaoxiansheng/Desktop/日常/数据集/orl_faces_full/*.pgm'
    elif number == 3:
        path = '/Users/zhuxiaoxiansheng/Desktop/日常/数据集/jaffe/*.tiff'
    elif number == 4:
        path = '/Volumes/TOSHIBA EXT/数据集/YaleB/*.pgm'
    
    pictures = io.ImageCollection(path)
    data = []
    for i in range(len(pictures)):
        picture = pictures[i]
        picture = skimage.color.rgb2gray(picture)
        data.append(np.ravel(picture.reshape((1,picture.shape[0]*picture.shape[1]))))
    label = []
    if number == 1:
        for i in range(len(data)):
            label.append(int(i/11))
    elif number == 2:
        for i in range(len(data)):
            label.append(int(i/10))
    elif number == 3:
        for i in range(len(data)):
            label.append(int(i/20))
    elif number == 4:
        label = [0]*64+[1]*64+[2]*64+[3]*64+[4]*64+[5]*64+[6]*64+[7]*64+[8]*64+[9]*64+[10]*60+[11]*59+[12]*60+[13]*63+[14]*62+[15]*63+[16]*63+[17]*64+[18]*64+[19]*64+[20]*64+[21]*64+[22]*64+[23]*64+[24]*64+[25]*64+[26]*64+[27]*64+[28]*64+[29]*64+[30]*64+[31]*64+[32]*64+[33]*64+[34]*64+[35]*64+[36]*64+[37]*64
    return np.matrix(data),np.matrix(label).T   


def SplitData(data,label,number,propotion):
    if number == 1:
        classes = 15
    elif number == 2:
        classes = 40
    elif  number == 3:
        classes = 10
    elif number == 4:
        trainData = []
        testData = []
        trainLabel = []
        testLabel = []
        lis = []
        while len(lis) < int(data.shape[0]*propotion):
            t = random.randint(0,data.shape[0]-1)
            if t not in lis:
                trainData.append(np.ravel(data[t,:]))
                trainLabel.append(np.ravel(label[t]))
                lis.append(t)
        for i in range(data.shape[0]):
            if i not in lis:
                testData.append(np.ravel(data[i,:]))
                testLabel.append(np.ravel(label[i]))
                lis.append(i)
        return np.matrix(trainData),np.matrix(trainLabel),np.matrix(testData),np.matrix(testLabel)  
        
    samples = data.shape[0]
    perClass = int(samples/classes)
    selected = int(perClass*propotion)
        
    trainData,testData = [],[]
    trainLabel,testLabel = [],[]
    count1 = []
    for i in range(classes):
        count2,k = [],math.inf
        for j in range(selected):
            count2.append(k)
            k = random.randint(0,perClass-1)
            while k in count2:
                k = random.randint(0,perClass-1)
            trainData.append(np.ravel(data[perClass*i+k]))
            trainLabel.append(np.ravel(label[perClass*i+k]))
            count1.append(11*i+k)
    for i in range(samples):
        if i not in count1:
            testData.append(np.ravel(data[i]))
            testLabel.append(np.ravel(label[i]))
    return np.mat(trainData),np.ravel(np.mat(trainLabel)),np.mat(testData),np.ravel(np.mat(testLabel)) 


#from sklearn.datasets import load_digits
#digits = load_digits()
#data = digits['data']
#label = digits['target']
#
#print(data.shape,max(label)+1)
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 1)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)
#
#
#from sklearn.datasets import load_breast_cancer
#breast_cancer = load_breast_cancer()
#data = breast_cancer['data']
#label = breast_cancer['target']
#
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 2018)


#dataset = 4
#data,label = LoadData(dataset)
#print(data.shape,max(label)+1)
#
#pca = PCA(0.99)
#data = pca.fit_transform(data) 
#data = data/255.
#
#def split(data,label,propotion):
#    train_index = np.random.choice(2414,size=(int(propotion*2414)),replace=False)
#    test_index = list(set(np.arange(2414))-set(train_index))
#    return data[train_index],label[train_index],data[test_index],label[test_index]
#
#
#traindata,trainlabel,testdata,testlabel = SplitData(data,label,dataset,0.8) 
#trainlabel = np.ravel(trainlabel)
#testlabel = np.ravel(testlabel)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)


#fault1_1 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase1.mat')['Set1_1']
#fault1_2 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase1.mat')['Set1_2']
#fault1_3 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase1.mat')['Set1_3']
#fault2_1 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase2.mat')['Set2_1']
#fault2_2 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase2.mat')['Set2_2']
#fault2_3 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase2.mat')['Set2_3']
#fault3_1 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase3.mat')['Set3_1']
#fault3_2 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase3.mat')['Set3_2']
#fault3_3 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase3.mat')['Set3_3']
#fault4_1 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase4.mat')['Set4_1']
#fault4_2 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase4.mat')['Set4_2']
#fault4_3 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase4.mat')['Set4_3']
#fault5_1 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase5.mat')['Set5_1'] 
#fault5_2 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase5.mat')['Set5_2']   
#fault6_1 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase6.mat')['Set6_1']
#fault6_2 = sio.loadmat(u'/Volumes/TOSHIBA EXT/数据集/CVACaseStudy/CVACaseStudy/FaultyCase6.mat')['Set6_2']
#
#data = np.vstack([fault1_1,fault1_2,fault1_3,fault2_1,fault2_2,fault2_3,fault3_1,fault3_2,fault3_3,fault4_1,fault4_2,fault4_3,fault5_1,fault5_2,fault6_1,fault6_2])
#label = [0]*(fault1_1.shape[0]+fault1_2.shape[0]+fault1_3.shape[0])+[1]*(fault2_1.shape[0]+fault2_2.shape[0]+fault2_3.shape[0])+[2]*(fault3_1.shape[0]+fault3_2.shape[0]+fault3_3.shape[0])+[3]*(fault4_1.shape[0]+fault4_2.shape[0]+fault4_3.shape[0])+[4]*(fault5_1.shape[0]+fault5_2.shape[0])+[5]*(fault6_1.shape[0]+fault6_2.shape[0])
#label = np.array(label)
#print(data.shape,max(label)+1)
#
#
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 2018)   
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)


#data = pd.read_csv(u'/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/dataset_diabetes/diabetic_data.csv')
#print(data.shape)
#
#le = preprocessing.LabelEncoder()
#for item in data.columns:
#    data[item] = le.fit_transform(data[item])
#label = data['diabetesMed'].values
#data = data.drop('diabetesMed',axis=1)
#data = data.drop('encounter_id',axis=1)
#data = data.drop('patient_nbr',axis=1)
#data = data.drop('weight',axis=1)
#data = data.drop('payer_code',axis=1)
#data = data.drop('max_glu_serum',axis=1)
#data = data.drop('A1Cresult',axis=1).values
#
#print(data.shape,max(label)+1)
#
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 2018)   
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)

#def Decode(Y_onehot):
#    Y = []
#    for i in range(Y_onehot.shape[0]):
#        lis = np.ravel(Y_onehot[i,:]).tolist()
#        Y.append(lis.index(max(lis)))
#    return np.array(Y)
#data = pd.read_excel(u'/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/Steel_Plates_Faults.xlsx')
#data = data.fillna(data.median())
#label = data.iloc[:,-7:].values
#label = Decode(label)
#data = data.drop([28,29,30,31,32,33,34],axis = 1)
#print(data.shape,label.shape)
#print(data.shape,max(label)+1)
#
#traindata,testdata,trainlabel,testlabel = train_test_split(data.values,label,test_size=0.2,random_state = 4)   
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)

#data = sio.loadmat('/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/USPS美国邮政服务手写数字识别库/USPStrainingdata.mat')
#traindata = data['traindata']
#traintarg = data['traintarg']
#trainlabel = Decode(traintarg)
#data = sio.loadmat('/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/USPS美国邮政服务手写数字识别库/USPStestingdata.mat')
#testdata = data['testdata']
#testtarg = data['testtarg']
#testlabel = Decode(testtarg)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)
#
#data = np.row_stack((traindata,testdata))
#label = np.ravel((np.row_stack((np.mat(trainlabel).T,np.mat(testlabel).T))))
#print(data.shape,max(label)+1)
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 2018)   
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)
#

#data = pd.read_excel(u'/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/car_evaluation.xlsx')
#
#le = preprocessing.LabelEncoder()
#for item in data.columns:
#    data[item] = le.fit_transform(data[item])
#label = data['label'].values
#data = data.drop('label',axis=1)
#data = data.values
#print(data.shape,max(label)+1)
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 1)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)


#data = pd.read_csv(u'/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/HTRU_2.csv')          
#label = data['label'].values
#data = data.drop('label',axis=1)
#data = data.values
#print(data.shape,max(label)+1)
#
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 2018)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)
#
#data = pd.read_excel(u'/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/mushroom_expand.xlsx')     
#
#le = preprocessing.LabelEncoder()
#for item in data.columns:
#    data[item] = le.fit_transform(data[item])
#    
#label = data['label'].values
#data = data.drop('label',axis=1)
#data = data.values
#print(data.shape,max(label)+1)
#
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 1)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)



#traindata = pd.read_csv('/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/Crowdsourced Mapping/training.csv')
#
#le = preprocessing.LabelEncoder()
#for item in traindata.columns:
#    traindata[item] = le.fit_transform(traindata[item])
#
#trainlabel = traindata['label'].values
#traindata = traindata.drop('label',axis = 1).values
#
#testdata = pd.read_csv('/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/Crowdsourced Mapping/testing.csv')
#
#le = preprocessing.LabelEncoder()
#for item in testdata.columns:
#    testdata[item] = le.fit_transform(testdata[item])
#
#testlabel = testdata['label'].values
#testdata = testdata.drop('label',axis = 1).values
#
#data = np.row_stack((traindata,testdata))
#label = np.ravel((np.row_stack((np.mat(trainlabel).T,np.mat(testlabel).T))))
#
#print(data.shape,max(label)+1)
#
#traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 1)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)
#



data = pd.read_excel(u'/Users/zhuxiaoxiansheng/Desktop/GBN/GBN_data/balance-scale.xlsx')  
  
le = preprocessing.LabelEncoder()
for item in data.columns:
    data[item] = le.fit_transform(data[item])


label = data['label'].values
data = data.drop('label',axis=1)
data = data.values
print(data.shape,max(label)+1)

traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.2,random_state = 0)
print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)


bls = broadnet(maptimes = 10, 
               enhencetimes = 10,
               map_function = 'relu',
               enhence_function = 'relu',
               batchsize = 100, 
               reg = 0.001)

starttime = datetime.datetime.now()
bls.fit(traindata,trainlabel)
endtime = datetime.datetime.now()
print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

predictlabel = bls.predict(testdata)
print(show_accuracy(predictlabel,testlabel))












