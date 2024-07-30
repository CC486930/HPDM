# -*- coding: utf-8 -*-

#train
import torch, time
from random import shuffle as sf
from torch import nn
from torch import optim
import numpy as np
from MyLoader import OrigDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from utils import inference, train, group_argtopk, writecsv, group_max, calc_err
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    best_acc = 0
    pk = 1 # Number of positive instances selected
    nk = 3 # Number of negatives instances selected
    n_epoch = 50 # Number of iterations
    test_every = 1 # Train n times test once
    # Defining the Network
    model = resnet18.resnet18(False)
    model.load_state_dict(torch.load('resnet18-5c106cde.pth'))# Weight files used for categorization
    model.fc = nn.Linear(model.fc.in_features, 2)
    #model.cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    cudnn.benchmark = True
    # Defining the data set
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    train_dset = XDataset('train-50.lib',transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=128,shuffle=False,
        pin_memory=False)
    test_dset = XDataset('test-50.lib',transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=128,shuffle=False,
        pin_memory=False)
    
    # fconv = open('Training_50.csv', 'w')
    # fconv.write('time,epoch,loss,error\n')
    # moment = time.time()
    # fconv.write('%d,0,0,0\n'%moment)
    # fconv.close()
    #
    # fconv = open('Testing_50.csv', 'w')
    # fconv.write('time,epoch,loss,error\n')
    # moment = time.time()
    # fconv.write('%d,0,0,0\n'%moment)
    # fconv.close()
    # Start Iteration
    for epoch in range(n_epoch):
        ## ①All Tests
        train_dset.setmode(1)
        _, probs = inference(epoch, train_loader, model, criterion)
#        torch.save(probs,'probs/train-%d.pth'%(epoch+1))
        probs1 = probs[:train_dset.plen] #plen is the number of tiles(probs) from positive instances in probs
        probs0 = probs[train_dset.plen:]

        print(train_dset.plen)
        ## ②Elected former pk=1
        topk1 = np.array(group_argtopk(np.array(train_dset.slideIDX[:train_dset.plen]), probs1, pk))
        ## ②Pick the first nk=5 and offset plen positions
        topk0 = np.array(group_argtopk(np.array(train_dset.slideIDX[train_dset.plen:]), probs0, nk))+train_dset.plen
        topk = np.append(topk1, topk0).tolist()
#        torch.save(topk,'topk/train-%d.pth'%(epoch+1))
#        maxs = group_max(np.array(train_dset.slideIDX), probs, len(train_dset.targets))
#        torch.save(maxs, 'maxs/%d.pth'%(epoch+1))
        sf(topk)
        ## ③Preparing the training set
        train_dset.maketraindata(topk)
        train_dset.setmode(2)
        ## ④Train and save the results
        loss, err = train(train_loader, model, criterion, optimizer)
        moment = time.time()
        writecsv([moment, epoch+1, loss, err], 'Training.csv')
        print('Training epoch=%d, loss=%.5f, error=%.5f'%(epoch+1, loss, err))
        ## ⑤validate
        if (epoch+1) % test_every == 0:
            test_dset.setmode(1)
            loss, probs = inference(epoch, test_loader, model, criterion)
#            torch.save(probs,'probs/test-%d.pth'%(epoch+1))
#            topk = group_argtopk(np.array(test_dset.slideIDX), probs, pk)
#            torch.save(topk, 'topk/test-%d.pth'%(epoch+1))
            maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))  #Returns the maximum probability of each slice
#            torch.save(maxs, 'maxs/test-%d.pth'%(epoch+1))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err = calc_err(pred, test_dset.targets)
            moment = time.time()
            writecsv([moment, epoch+1, loss, err], 'Testing.csv')
            print('Testing epoch=%d, loss=%.5f, error=%.5f'%(epoch+1, loss, err))
            #Save best model
        if 1-err >= best_acc:
            best_acc = 1-err
            obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                    }
            torch.save(obj, 'checkpoint_50_best.pth')
            
if __name__ == '__main__':
    main()
