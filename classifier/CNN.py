from pickletools import optimize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc, fbeta_score, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
# from tensorflow.python.keras.backend import dropout, relu
# from tensorflow.python.keras.layers.merge import concatenate
import utils.tools as utils
from keras.constraints import maxnorm
from keras.models import Sequential, Input
from keras.layers import Dense
from keras.layers import concatenate
from keras.layers import Flatten
import numpy as np
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model

import torch
from torch import conv1d, nn
import torch.nn.functional as F

#bilstm:hiddens=100,output=(1000,1,200)
#attention: output=(1000,1,272)

net = nn.Sequential(
    nn.Conv1d(472,64,kernel_size=3),nn.ReLU(),
    nn.Conv1d(64,32,kernel_size=3),nn.ReLU(),
    nn.Conv1d(32,16,kernel_size=3),nn.ReLU(),
    nn.MaxPooling1D(kernel_size=3),
    nn.Flatten(),
    nn.Linear(16,2)
)

#data

#train
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop()

#627??
left_input = Input(shape=((464, 1)))
left_conv = Conv1D(64, 3, padding='same', activation='relu')(left_input)
left_conv = Conv1D(32, 3, padding='same', activation='relu')(left_conv)
left_conv = Conv1D(16, 3, padding='same', activation='relu')(left_conv)
left_conv = MaxPooling1D()(left_conv)
left_conv = Flatten()(left_conv)
layer_out = Dense(2, activation='sigmoid')(left_conv)
model = Model(inputs=left_input, outputs=layer_out)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.summary()

data_=pd.read_csv(r'input file',header=None)
data=np.array(data_)
data=data[:,0:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
#label1=np.ones((544,1))#Value can be changed
#label2=np.zeros((407,1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
y=label
sepscores = []

ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

X = X.reshape(m1, n1,1)

#model.fit([X,X], [y,y], epochs=10)

skf= StratifiedKFold(n_splits=5)

for train, test in skf.split(X,y):
    y_train=utils.to_categorical(y[train])#generate the resonable results
    cv_clf = model
    hist=cv_clf.fit([X[train],X[train]],
                    [y_train,y_train],
                    epochs=10)
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    y_score=cv_clf.predict(X[test])#the output of  probability
    yscore=np.vstack((yscore,y_score))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('GTB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
    #hist=[]
    #cv_clf=[]
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
#yscore_sum.to_csv('yscore_imCNN_SBP_test.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
#ytest_sum.to_csv('ytest_imCNN_SBP_test.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='DL_1 ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv(r'save file')

