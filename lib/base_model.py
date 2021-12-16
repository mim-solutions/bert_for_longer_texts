import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class Model():
    def __init__(self):
        self.params = None
        self.preprocessor = None
        self.dataset_class = None
        self.collate_fn = None
    def create_dataset(self,X_preprocessed,y):
        dataset = self.dataset_class(X_preprocessed,y)
        return dataset
    def fit(self, X_train, y_train, epochs = 3):
        train_samples = len(X_train)
        # Text preprocessing
        X_train_preprocessed = self.preprocessor.preprocess(X_train)
        # Creating train dataset
        train_dataset = self.create_dataset(X_train_preprocessed,y_train)
        # Creating train dataloader
        train_dataloader = create_train_dataloader(train_dataset,self.params['batch_size'],self.collate_fn)
        # Training
        self.train_for_epochs(train_samples,train_dataloader,epochs)
    def predict(self, X_test):
        test_samples = len(X_test)
        y_test = [0] * len(X_test) # dummy labels
        # Text preprocessing
        X_test_preprocessed = self.preprocessor.preprocess(X_test)
        # Creating test dataset
        test_dataset = self.create_dataset(X_test_preprocessed,y_test)
        # Creating train dataloader
        test_dataloader = create_test_dataloader(test_dataset,self.params['batch_size'],self.collate_fn)
        # Prediction
        _, _, preds, _ = self.evaluate_single_epoch(test_samples,test_dataloader)
        return preds
    def evaluate_single_batch(self,batch,model,device):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        model_input = batch[:-1]

        labels = batch[-1]

        # model predictions
        preds = model(*model_input)
        preds = torch.flatten(preds).cpu()
        labels = labels.float().cpu()
        return preds, labels

    def train_single_epoch(self,train_samples,train_dataloader):
        model = self.nn
        model.train()
        
        total_loss = 0
        total_accurate = 0

        # iterate over batches
        for step,batch in enumerate(train_dataloader):
             
            # zero the parameter gradients
            self.optimizer.zero_grad()

            preds, labels = self.evaluate_single_batch(batch,model,self.params['device'])

            # compute the loss between actual and predicted values
            loss,total_accurate,total_loss = calc_loss_and_accuracy(preds,labels,total_loss,total_accurate)

            # backward pass to calculate the gradients
            loss.backward()

            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

        # compute the train loss of the epoch
        avg_loss = total_loss / train_samples
        accuracy = total_accurate/ train_samples
        return avg_loss, accuracy
    
    def train_for_epochs(self, train_samples, train_dataloader, epochs = 3):
        for epoch in range(epochs):
            avg_loss, accuracy = self.train_single_epoch(train_samples,train_dataloader)
            print(f'Epoch: {epoch}, Train accuracy: {accuracy}, Train loss: {avg_loss}')
            
    def evaluate_single_epoch(self, test_samples, test_dataloader):
        model = self.nn
        total_loss = 0
        total_accurate = 0

        preds_total = []
        labels_total = []

        # deactivate dropout layers
        model.eval()

        # iterate over batches
        for step,batch in enumerate(test_dataloader):

            # deactivate autograd
            with torch.no_grad():

                preds, labels = self.evaluate_single_batch(batch,model,self.params['device']) 
                preds_total.extend(preds)
                labels_total.extend(labels)           

                # compute the validation loss between actual and predicted values
                
                loss,total_accurate,total_loss = calc_loss_and_accuracy(preds,labels,total_loss,total_accurate)

        # compute the evaluation loss of the epoch
        preds_total = [x.item() for x in preds_total]
        labels_total = [x.item() for x in labels_total]
        avg_loss = total_loss / test_samples
        accuracy = total_accurate/ test_samples
        return avg_loss, accuracy, preds_total,labels_total



def create_dataloader(data,sampler_class,batch_size,collate_fn = None):
    sampler = sampler_class(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, collate_fn = collate_fn)
    return dataloader

def create_train_dataloader(train_data,batch_size,collate_fn = None):
    train_dataloader = create_dataloader(train_data,RandomSampler,batch_size,collate_fn)
    return train_dataloader

def create_test_dataloader(test_data,batch_size,collate_fn = None):
    test_dataloader = create_dataloader(test_data,SequentialSampler,batch_size,collate_fn)
    return test_dataloader

def create_dataloaders(train_data,test_data, batch_size, collate_fn = None):

    train_dataloader = create_train_dataloader(train_data,batch_size,collate_fn)
    test_dataloader = create_test_dataloader(test_data,batch_size,collate_fn)

    return train_dataloader, test_dataloader



def calc_loss_and_accuracy(preds,labels,total_loss,total_accurate):
    cross_entropy = nn.BCELoss(reduction = 'sum')
    loss = cross_entropy(preds, labels)
    
    total_loss = total_loss + loss.detach().cpu().numpy()
    
    predicted_classes = (preds.detach().numpy() >= 0.5)

    accurate = sum(predicted_classes == np.array(labels).astype(bool))
        
    total_accurate = total_accurate + accurate

    return loss,total_accurate,total_loss