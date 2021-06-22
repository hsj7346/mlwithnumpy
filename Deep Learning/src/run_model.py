import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def stop_condition(stop_thr, before, after):
  if before == 0 and after == 0:
    return False
  elif stop_thr > abs(after - before):
    return True
  else:
    return False

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
    loss = {}
    loss['train'] = []
    loss['valid'] = []
    accuracy = {}
    accuracy['train'] = []
    accuracy['valid'] = []
    if train_set == None:
      pass
    else:
      trainin = DataLoader(train_set, batch_size, shuffle)
    if valid_set == None:
      pass
    else:
      vali = DataLoader(valid_set, shuffle = shuffle, batch_size = batch_size)
    if test_set == None:
      pass
    else:
      test = DataLoader(test_set, shuffle = shuffle, batch_size = batch_size)
    if running_mode == 'train':
      optimizer = optim.SGD(model.parameters(),lr = learning_rate)
      if (valid_set != None):
        before = 0
        after = 0
        n = 0
        while ((stop_condition(stop_thr, before, after) == False) and (n < n_epochs)):
          before = after
          model1, t_loss, t_acc = _train(model,trainin,optimizer)
          after, v_acc = _test(model1,vali)
          loss['train'].append(t_loss)
          loss['valid'].append(after)
          accuracy['train'].append(t_acc)
          accuracy['valid'].append(v_acc)
          n += 1
      else:
        n = 0
        while (n < n_epochs):
          _train(model,trainin,optimizer)
          n += 1
      return model, loss, accuracy
    else:
      return _test(model,test)

      


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    criter = nn.CrossEntropyLoss()
    train_loss = 0
    total = 0
    correct = 0
    correct_tot = 0
    model.train()
    for _, data in enumerate(data_loader,0):
      features,labels = data
      optimizer.zero_grad()
      outputs = model(features.float())
      loss = criter(outputs,labels.long())
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      total += 1
      _, predicted = torch.max(outputs.data, 1)
      correct += (predicted == labels).sum().item()
      correct_tot += labels.size(0)
    return model, train_loss/total, correct/correct_tot * 100



def _test(model, data_loader, device=torch.device('cpu')):
    criter = nn.CrossEntropyLoss()
    test_loss = 0
    total = 0
    correct = 0
    correct_tot = 0
    model.eval()
    for _, data in enumerate(data_loader,0):
      features,labels = data
      outputs = model(features.float())
      loss = criter(outputs,labels.long())
      test_loss += loss.item()
      total += 1
      _, predicted = torch.max(outputs.data, 1)
      correct += (predicted == labels).sum().item()
      correct_tot += labels.size(0)
    return test_loss/total, correct/correct_tot * 100

