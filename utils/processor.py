import torch
import numpy as np
from tqdm import tqdm
import optuna
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn
import os
from datetime import datetime
import time
from .metrics import get_mean_dist, get_param_count_BraggNN, get_param_count_Deepsets, get_inference_time, get_acc
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Trains BraggNN models and calculates all metrics 
def evaluate_BraggNN(model, train_loader, val_loader, device, num_epochs = 50, lr = .0015, weight_decay=2.2e-9):
    model = model.to(device)

    #Train Model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    validation_loss = train(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)
    
    #Evaluate Performance
    mean_distance = get_mean_dist(model, val_loader, device)
    #Evaluate Efficiency
    param_count = get_param_count_BraggNN(model) #Just for reference, we are not optimizing for this. We measure BOPs in global_search.py
    inference_time = get_inference_time(model, device, img_size=(256,1,11,11)) #Just for reference, we are not optimizing for this.

    print('Mean Distance: ', mean_distance, ', Inference time: ', inference_time, ', Validation Loss: ', validation_loss, ', Param Count: ', param_count)
    return mean_distance, inference_time, validation_loss, param_count

#Trains model and calculates all metrics 
def evaluate_Deepsets(model, train_loader, val_loader, device, num_epochs = 80, lr = .0032):
    model = model.to(device)
    
    #Train Model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    validation_loss = train(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs, patience=7)
    
    #Evaluate Performance
    acc = get_acc(model, val_loader, device)

    #Evaluate Efficiency
    param_count = get_param_count_Deepsets(model)
    inference_time = get_inference_time(model, device, img_size=(1024,3,8)) #Just for reference, we are not optimizing for this.

    print('Accuracy: ', acc, ', Inference time: ', inference_time, ', Validation Loss: ', validation_loss, ', Param Count: ', param_count)
    return acc, inference_time, validation_loss, param_count

def train(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, num_epochs, patience=5):
    curr_patience = patience
    previous_epoch_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device).float(), targets.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_loss += loss.item()

        validation_loss /= len(valid_loader)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(validation_loss)
        else:
            scheduler.step()

        # Early Stopping Procedure
        if validation_loss < previous_epoch_loss:
            curr_patience=patience
        else:
            curr_patience -= 1
            if curr_patience <= 0: break
        previous_epoch_loss = validation_loss

    return previous_epoch_loss
