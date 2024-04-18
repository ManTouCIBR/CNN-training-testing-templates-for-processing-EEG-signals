import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader,Dataset
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torchvision.datasets
from sklearn.preprocessing import MinMaxScaler
import mne,glob,os,re,torch,sklearn,warnings
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import torch
import torchvision.transforms as transforms
transf = transforms.ToTensor()
from sklearn.preprocessing import StandardScaler
from EEGCNN_Module import *
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn 
from sklearn import metrics
import logging
from scipy import signal

#帮助类
class Config:
    batch_size = 32
    lr = 0.01 
    epochs = 800

def buttferfiter(data):
    Fs = 250
    b, a = signal.butter(4, [8, 30], 'bandpass',fs=Fs)
    data = signal.filtfilt(b, a, data, axis=1)
    return data

class feature_dataset(Dataset):
    def __init__(self,file_path,target_path,transform =None):
        self.file_path = file_path
        self.target_path = target_path

        self.data = self.parse_data_file(file_path)
        self.target = self.parse_target_file(target_path)
        self.transform = transform

    def parse_data_file(self,file_path):   
        data = pd.read_csv(file_path,header=None)
        data = np.array(data,dtype=np.float32)
        #data =  buttferfiter(data)
        data = torch.tensor(data)
        num_sub = len(data)
        scaler =StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data.reshape(num_sub,22,1000)
        data = torch.tensor(data)

        data = data.unsqueeze(1)
        '''
        for row in data:
          row = row.strip(" ").split(' ')
          row = np.array([np.float32(cell) for cell in row])
          #data = torch.ternsor(row)
        '''
        return np.array(data,dtype=np.float32)

    def parse_target_file(self,target_path):
            
        target = pd.read_csv(target_path,header=None)
        target = np.array(target,dtype=np.float32)
        encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
        x_hot = np.array(target)
        x_hot = x_hot.reshape(-1,1)
        encoder.fit(x_hot)
        x_oh = encoder.transform(x_hot).toarray()
        print(x_oh,x_oh.shape)
        d=transf(x_oh)  
        x_hot_label = torch.argmax(d, dim=2).long()
        print(x_hot_label.shape)
        label = x_hot_label
        label = label.transpose(1,0)
        target = torch.squeeze(label)
        return target

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, index):
        x = self.data[index]
        #index2 = np.random.choice(len(self.data))
        #x2 = self.data[index2,:]
        target = self.target[index]
         
        return x,target 

def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 定义L1正则化函数
def l1_regularizer(weight, lambda_l1):
    return lambda_l1 * torch.norm(weight, 1)

# 定义L2正则化函数
def l2_regularizer(weight, lambda_l2):
    return lambda_l2 * torch.norm(weight, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
config = Config()

train_transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = feature_dataset(file_path = r'C:\Users\19067\Desktop\EEG-TCNet\MI-EEG-A01T.csv',
                                target_path=r'C:\Users\19067\Desktop\EEG-TCNet\EtiquetasA01T.csv')
train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=config.batch_size)

val_transforms = transforms.Compose([transforms.ToTensor()])
val_dataset = feature_dataset(file_path =r'C:\Users\19067\Desktop\EEG-TCNet\MI-EEG-A01E.csv',
                               target_path=r'C:\Users\19067\Desktop\EEG-TCNet\EtiquetasA01E.csv')
val_dataloader = DataLoader(val_dataset,batch_size=config.batch_size)  
set_seed()

model = EEG_TCNet().to(device)
optimizer = optim.SGD(model.parameters(),lr=config.lr,weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
loss_fn = criterion.to(device) 
 
def show_plot(accuracy_history,loss_history,test_accuracy):
    plt.figure(figsize=(20,10))
    #fig2
    plt.subplot(121)
    plt.plot(loss_history,marker=".",color="c")
    plt.title('train loss')
    #fig3
    plt.subplot(122)
    plt.plot(accuracy_history,marker="o",label="train_acc") #plt.plot(x,y)定义x，y轴数据，定义颜色，标记型号，大小等
    plt.plot(test_accuracy, marker='o', label="test_acc")
    plt.title("ACC")
    plt.legend(loc="best")
    plt.savefig('acc_loss.png')
    plt.show()
    
def plot_recall(epoch_list,recall1,recall2,recall3,recall4):
    plt.figure(figsize=(15,8)) 
    plt.plot(epoch_list,recall1, color='purple', label='Back1_Recall',marker=".")
    plt.plot(epoch_list,recall2,color='c',label="Back2_Recall",marker=".")
    plt.plot(epoch_list,recall3,color='g',label="Back3_Recall",marker=".")
    plt.plot(epoch_list,recall4,color='m',label="Back4_Recall",marker=".")
    plt.title('Recall during test')
    plt.xlabel('Epoch')
    plt.ylabel('Recall_Vales')
    plt.legend()
    plt.savefig("recall.jpg")
    plt.show()

def plot_precision(epoch_list,precision1,precision2,precision3,precision4):
    plt.figure(figsize=(15,8))
    plt.plot(epoch_list,precision1, color='black', label='Back1_Precision',marker="o")
    plt.plot(epoch_list,precision2, color='b', label='Back2_Precision',marker="o")
    plt.plot(epoch_list,precision3, color='m', label='Back3_Precision',marker="o")
    plt.plot(epoch_list,precision4, color='c', label='Back4_Precision',marker="o")
    plt.xlabel('Epoch')
    plt.ylabel('Precision_Vales')
    plt.title('Precision during test')
    plt.legend()
    plt.savefig("precision.jpg")
    
    plt.show()

def plot_f1(epoch_list,f1_1,f1_2,f1_3,f1_4):
    plt.figure(figsize=(15,8))
    plt.plot(epoch_list,f1_1, color='yellow', label='Back1_F1',marker="^")
    plt.plot(epoch_list,f1_2, color='g', label='Back2_F1',marker="^")
    plt.plot(epoch_list,f1_3, color='b', label='Back3_F1',marker="^")
    plt.plot(epoch_list,f1_4, color='m', label='Back4_F1',marker="^")
    plt.xlabel('Epoch')
    plt.ylabel('F1_Values')
    plt.title('f1 during test')
    plt.legend()
    plt.savefig("f1.jpg")
    plt.show()
    
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
    
def DrawConfusionMatrix(save_model_name,val_dataloader):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEG_TCNet().to(device)
    model.load_state_dict(torch.load(os.path.join(save_model_name,"EEG_TCNet.pth")))
    model.eval()
    predict = []
    gt = []
    with torch.no_grad():
        for data_label in val_dataloader:
            x,target = data_label
            x,target = x.to(device),target.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            tmp_predict = predicted.cpu().detach().numpy()
            tmp_label = target.cpu().detach().numpy()

            if len(predict) == 0:
                predict = np.copy(tmp_predict)
                gt = np.copy(tmp_label)
            else:
                predict = np.hstack((predict,tmp_predict))
                gt = np.hstack((gt,tmp_label))

    cm = confusion_matrix(y_true=gt, y_pred=predict)   # 混淆矩阵
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = ['left','right','feet',"tongue"])
    disp.plot()
    save_hunxiao_path = os.path.join(save_model_name,'混淆矩阵.png')
    plt.savefig(save_hunxiao_path,dpi = 1000)

recall1= []
recall2= []
recall3 = []
recall4 = []
precision1 = []
precision2 = []
precision3 = []
precision4 = []
f1_1 = []
f1_2 = []
f1_3 = []
f1_4 = []
epoch_list = []    
accuracy_history = []
loss_history = []
test_accuracy = []
best_acc = 0

# logger = get_logger(os.path.join(r"C:\Users\19067\Desktop\EEGCNN-PYTHONS",'all_trail_exp.log'))

for epoch in range(0,Config.epochs):
    model.train()
    counter = []
    iteration_number = 0
    train_correct = 0
    total = 0
    correct = 0
    train_loss = 0
    for i,data in enumerate(train_dataloader,0): #enumerate防止重复抽取到相同数据，数据取完就可以结束一个epoch
        data,label = data
        #data = np.copy(data)
        data,label= data.to(device),label.to(device)
        optimizer.zero_grad() 
        output = model(data)
        
        #lambda_l2 = 0.0001  
        #l2_regularization = l2_regularizer(model.weight(), lambda_l2)
        
        loss = loss_fn(output,label) 
        #loss += l2_regularization
        
        loss.backward()   
        optimizer.step()  
        predicted=torch.argmax(output, 1)
        train_correct += (predicted == label).sum().item()
        total+=label.size(0) 
        
        train_loss += loss
    train_accuracy = train_correct / total
    train_loss /= len(train_dataloader)
    train_loss = train_loss.item()
    iteration_number += 1
    
    counter.append(iteration_number)
    accuracy_history.append(train_accuracy)
    loss_history.append(train_loss)
    
    # print("Epoch number {}\n Current Train  Accuracy {}\n Current Train loss {}\n".format
    #         (epoch, train_accuracy,train_loss))
    # logger.info("Epoch number {}\n Current Train  Accuracy {}\n Current Train loss {}\n".format
    #         (epoch, train_accuracy,train_loss))
    
    with torch.no_grad():
        model.eval()
        test_correct = 0
        total =  0
        tensor_concat_pre_label = []
        label_item = []
        epoch_recall = 0
        epoch_precision = 0
        epoch_f1 = 0
        n_classes = 4
        target_num = torch.zeros((1, n_classes)) 
        predict_num = torch.zeros((1, n_classes))
        acc_num = torch.zeros((1, n_classes))

        for idx, data in enumerate(val_dataloader,0):
            data,label = data
            data,label = data.to(device),label.to(device)
            output = model(data)
            predicted=torch.argmax(output, 1)
            test_correct += (predicted == label).sum().item()
            total+=label.size(0)
            # 1 PR/RE/F1 报告
            pred = predicted
            y_true = label.cpu()
            y_pred = pred.float().cpu()
            if len(tensor_concat_pre_label)==0:
                tensor_concat_pre_label = y_pred.clone()
                label_item = y_true.clone()
            else:
                tensor_concat_pre_label = torch.concat((tensor_concat_pre_label,y_pred))
                label_item = torch.concat((label_item,y_true)) 
        sklearn.metrics.classification_report(label_item,tensor_concat_pre_label)
        sklearn.metrics.accuracy_score(label_item, tensor_concat_pre_label)
        print(accuracy_score(label_item,tensor_concat_pre_label),classification_report(label_item,tensor_concat_pre_label))
        print(metrics.confusion_matrix(label_item,tensor_concat_pre_label))
        
        # logger.info(accuracy_score(label_item,tensor_concat_pre_label))
        #2 acc
        current_test_acc = test_correct / total
        test_accuracy.append(current_test_acc)
        print("测试acc: ",(current_test_acc) * 100,"%")
        #3 每一类别的pr、re、f1图
        pre_mask = torch.zeros(output.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
        
        tar_mask = torch.zeros(output.size()).scatter_(1, label.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)  # 得到数据中每类的数量
        
        acc_mask = pre_mask * tar_mask 
        acc_num += acc_mask.sum(0) # 得到各类别分类正确的样本数量

    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    recall  = recall.numpy()
    precision = precision.numpy()
    F1 = F1.numpy()
    recall_back1,recall_back2,recall_back3,recall_back4 = recall[:,0],recall[:,1],recall[:,2],recall[:,3]
    precision_back1,precision_back2,precision_back3,precision_back4 = precision[:,0],precision[:,1],precision[:,2],precision[:,3]
    F1_back1,F1_back2,F1_back3,F1_back4 = F1[:,0],F1[:,1],F1[:,2],F1[:,3]
    #accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
    epoch_list.append(epoch)
    recall1.append(recall_back1)
    recall2.append(recall_back2)
    recall3.append(recall_back3)
    recall4.append(recall_back4)
    precision1.append(precision_back1)
    precision2.append(precision_back2)
    precision3.append(precision_back3)
    precision4.append(precision_back4)
    f1_1.append(F1_back1)  
    f1_2.append(F1_back2)
    f1_3.append(F1_back3)
    f1_4.append(F1_back4)

    # if current_test_acc > best_acc and epoch>Config.epochs/2:
    best_acc = current_test_acc
    torch.save(model.state_dict(),"EEG_TCNet.pth")
    
DrawConfusionMatrix(r"C:\Users\19067\Desktop\EEG-TCNet",val_dataloader)
 
plot_recall(epoch_list,recall1,recall2,recall3,recall4)
plot_precision(epoch_list,precision1,precision2,precision3,precision4)
plot_f1(epoch_list,f1_1,f1_2,f1_3,f1_4)
show_plot(accuracy_history,loss_history,test_accuracy) 

import torchvision.models as models
from torchsummary import summary
summary(model,(1,22,1000),batch_size=32,device="cuda")
print(model)
