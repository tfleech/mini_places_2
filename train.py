import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True
import numpy as np

import dataset
from models.AlexNet import *
from models.ResNet import *

def run():
    # Parameters
    num_epochs = 10
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model_sgd.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here

        gc.collect()
        epoch += 1 
    return


def run_test():
    batch_size = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model.load_state_dict(torch.load('./models/model.10', map_location=device))
    model = model.to(device)

    val_loader, test_loader = dataset.get_val_test_loaders(batch_size)
    num_test_batches = len(val_loader)

    top_5_correct = 0
    top_1_correct = 0
    total = 0

    for batch_num, (inputs, labels) in enumerate(val_loader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        #print(inputs)

        with torch.no_grad():
            output = model(inputs)[0,:]
            output = output.to('cpu')
            output_sorted = output.numpy().argsort()
        #print(output[output_sorted[0]])
        #print(output[output_sorted[-1]])
        top_5 = output_sorted[-5:]
        for i in top_5:
            if i == int(labels):
                top_5_correct += 1
        if int(torch.argmax(output)) == int(labels):
            top_1_correct += 1
        total += 1
        if total%100 == 0:
            print(total)
        if total > 1000:
            print("Top 5 correct: ", float(top_5_correct)/total)
            print("Top 1 correct: ", float(top_1_correct)/total)
            return


print('Starting training')
run()
print('Training terminated')
