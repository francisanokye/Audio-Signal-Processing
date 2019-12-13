import os
import glob       
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import models
from torch import optim

csv_info = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
csv_info = csv_info.set_index('slice_file_name')

# wrapper class for the UrbanSound8K dataset
class AudioDataset(Dataset):
    """
    A rapper class for the UrbanSound8K dataset.
    """
    def __init__(self, file_path, audio_paths, folds):
        """
        Args:
            file_path(string): path to the audio csv file
            root_dir(string): directory with all the audio folds
            folds: integer corresponding to audio fold number or list of fold number if more than one fold is needed
        """
        self.audio_file = pd.read_csv(file_path)
        self.folds = folds
        self.audio_paths = glob.glob(audio_paths + '/*' + str(self.folds) + '/*')
    
    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        
        audio_path = self.audio_paths[idx]
        audio, rate = torchaudio.load(audio_path, normalization=True)
        audio = audio.mean(0, keepdim=True)
        c, n = audio.shape
        zero_need = 160000 - n
        audio_new = F.pad(audio, (zero_need //2, zero_need //2), 'constant', 0)
        audio_new = audio_new[:,::5]
        
        #Getting the corresponding label
        audio_name = audio_path.split(sep='/')[-1]
        labels = self.audio_file.loc[self.audio_file.slice_file_name == audio_name].iloc[0,-2]
        
        return audio_new, labels

# visualization function        
import matplotlib.pyplot as plt

def display(acc_train,acc_test,loss_test,loss_train,model,crossVal):
    
    """Display the average loss and accuracy for the train and the Test"""
    
    plt.figure(figsize=(12, 7))
    plt.subplot(121)
    plt.plot(acc_train, label='Train accuracy')
    plt.plot(acc_test, label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.title('model' +' Train CrossValidation '+str(crossVal))

    plt.subplot(122)
    plt.plot(loss_train, label='Train loss')
    plt.plot(loss_test, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.legend()
    
    plt.title('model' +' Test CrossValidation '+str(crossVal))
    plt.show()

# accuracy function
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



# Train_test_with_CrossValidation
import numpy as np
def train_test_with_CrossValidation(file_path,audio_paths,model,criterion,epochs):
    """args:trainSet,testSet, model and the criteron to compute and the number of epochs
         Compute the average loss and accuracy of the train and the test
    """
    for i in range(1,11):
        #cross validation
        folds = [1,2,3,4,5,6,7,8,9,10]
        test_set =AudioDataset(file_path,audio_paths,folds[i-1])
        folds.remove(i)
        train_set = AudioDataset(file_path,audio_paths,folds)
        #print('CrossValidation {} : '.format(i))
        #print("Train set size: " + str(len(train_set)))
        #print("Test set size: " + str(len(test_set)))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = False,num_workers=10)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = False, num_workers=10)
        
        #train and Test
        optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
        acc_train = []
        acc_test = []
        model.to(device)
        
        loss_train = []
        loss_test = []
        mean_top = []
        mean_loss_all = []
       # wandb.init(project="audio_classifier_task-")
        # loop over the dataset multiple times
        for epoch in range(epochs):

            #print('Epoch: {}'.format(epoch))

            # train for 1 epoch on the train set
            correct = 0
            l_loss = []

            for j, (batch, targets) in enumerate(train_loader):

                # batch and targets to cuda 
                batch, targets = batch.to(device), targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward 
                outputs = model(batch)
                loss = criterion(outputs,targets)
                l_loss.append(loss.item())
                # backward
                loss.backward()
                # gradient step 
                optimizer.step()
                # compute accuracy wandb.log({'loss': 0.2}, step=step)
                correct += accuracy(outputs, targets)[0].item()
                   
            scheduler.step()
            top_1 = 100 * correct / len(train_loader.dataset)
            acc_train.append(top_1)
            
            sum_loss = np.mean(l_loss)
            loss_train.append(sum_loss)

            #wandb.log({"Train Accuracy":top_1, "Train Loss": sum_loss})
    

            # evaluate on the test set 

            with torch.no_grad():
                correct = 0
                l_loss = []
                for j, (batch,targets) in enumerate(test_loader):

                    # batch and targets to cuda 
                    batch, targets = batch.to(device), targets.to(device)

                    # forward 
                    outputs = model(batch)
                    loss = criterion(outputs,targets)
                    l_loss.append(loss.item())
                    # compute accuracy 
                    correct += accuracy(outputs, targets)[0].item()

            top_1 =100 * correct / len(test_loader.dataset)        
            acc_test.append(top_1)
            
            sum_loss = np.mean(l_loss)
            loss_test.append(sum_loss)
            #wandb.log({"Test Accuracy":top_1, "Test Loss": sum_loss})
          
        print('Train accuracy: {:.2f}%'.format(np.mean(acc_train)))
        print('Train loss: {:.2f}'.format(np.mean(loss_train)))
        print('**************')   
        print('Test accuracy: {:.2f}%'.format(np.mean(acc_test)))
        print('Test loss: {:.2f}'.format(np.mean(loss_test)))   
        display(acc_train,acc_test,loss_test,loss_train,model,i)
      
#train_test_model(train_loader,test_loader,model,criterion,epochs)

# fefine criterion, file_path, audio_paths and epochs
criterion = nn.CrossEntropyLoss()
file_path = '/content/drive/My Drive/UrbanSound8K/metadata/UrbanSound8K.csv'
audio_paths = '/content/drive/My Drive/UrbanSound8K/audio'
epochs = 10
device = 'cuda:0'

def init_weights(m):
   if type(m) == nn.Conv1d or type(m) == nn.Linear:
       nn.init.xavier_uniform_(m.weight.data)

# Train for all models
model3 = M3_Model()
model3.to(device)
model3.apply(init_weights)

model5 = M5_Model()
model5.to(device)
model5.apply(init_weights)

model11 = M11_Model()
model11.to(device)
model11.apply(init_weights)

model18 = M18_Model()
model18.to(device)
model18.apply(init_weights)

modelRes = ResNet()
modelRes.to(device)
modelRes.apply(init_weights)

models_name = ['M3','M5','M11','M18','M34-Res']
models = [model3,model5,model11,model18,modelRes]
#wandb.init(project="audio_classifier_task-")
for name,model in zip(models_name,models):
  #wandb.watch(model,log = 'all')
  #model.train()
  print('Model : {}'.format(name))
  train_test_with_CrossValidation(file_path,audio_paths,model,criterion,epochs)

  #print(model)

    