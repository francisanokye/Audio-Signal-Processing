

import numpy as np
import AudioDataset
import models
import loaddata
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train_test_with_CrossValidation(file_path,audio_paths,model,criterion,epochs):
    """args:
    trainSet,testSet, model and the criteron to compute and the number of epochs
    Compute the average loss and accuracy of the train and the test
    """
    for i in range(1,11):
        #cross validation
        folds = [1,2,3,4,5,6,7,8,9,10]
        test_set = AudioDataset(file_path,audio_paths,folds[i-1])
        folds.remove(i)
        train_set = AudioDataset(file_path,audio_paths,folds)
        #print('CrossValidation {} : '.format(i))
        #print("Train set size: " + str(len(train_set)))
        #print("Test set size: " + str(len(test_set)))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = False,num_workers=10)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = False, num_workers=10)
        
        #train and Test
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
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

            #wandb.log({"Train Accuracy":top_1, "Train Loss": sum_loss}) #to visualize the train and test during training

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
      
# train model
#train_test_model(train_loader,test_loader,model,criterion,epochs)

# plot the loss and accuracy after training and testing
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
                        