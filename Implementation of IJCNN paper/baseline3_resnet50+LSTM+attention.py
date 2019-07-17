import torch
import resnet1 as resnet
import numpy as np
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import time
import shutil

import os
import torch
import torch.utils.data as data
from PIL import Image
cuda0 = torch.device('cuda:0')
def default_loader(path):
    return Image.open(path).convert('RGB').resize((224,224))

class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform = None, target_transform=None, loader=default_loader):
        fh = open(label)
        c=0
        imgs=[]
        class_names='label'
        for line in  fh.readlines():
            cls = line.rsplit('\t', 1)
            fn = cls.pop(0)
            if os.path.isfile(os.path.join(root, fn)):
                imgs.append((fn, int(cls[0])))
            c=c+1
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label-1,dtype=torch.long)

    def __len__(self):
        return len(self.imgs)
    
    def getName(self):
        return self.classes

class Resnet50_MLP(nn.Module):

    def __init__(self):
        super(Resnet50_MLP, self).__init__()
        self.resnet50 = resnet.resnet50(pretrained=True)
        #self.lstm = nn.LSTM(input_dim, output_dim)
        # The linear layer that maps from hidden state space to tag space
        self.mlp1 = nn.Linear(1000, 256)
        self.mlp2 = nn.Linear(256, 23)
        self.lstm = nn.LSTM(2048*7, 256,batch_first=True)
        self.attention = nn.Linear(256,1)

    def forward(self, data):
        x = self.resnet50(data)
        x = torch.cat([torch.cat([x[i,j,:,:] for j in range(2048)],dim=1).view(1,7,2048*7) for i in range(64)],dim=0)
        #for i in range(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out
        L = F.tanh(x)
        attention = F.softmax(self.attention(L),dim=1)
        x = torch.matmul(torch.transpose(attention,dim0=1,dim1=2),x).squeeze()
        #x = self.mlp1(x)
        x = F.tanh(x)
        x = self.mlp2(x)
        return x

def main():
    # apply the resnet 50
    resnet50 = Resnet50_MLP()#.to(cuda0)
    """
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    """
    #cudnn.benchmark = True

    # Data loading code
    #traindir = './train'
    #valdir = './val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    a = myImageFloder(root = "./chromosomes_data/train", label = "./chromosomes_data/all_label.txt", 
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    train_loader = torch.utils.data.DataLoader(
            myImageFloder(root = "./chromosomes_data/train", label = "./chromosomes_data/all_label.txt", 
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])), 
            batch_size= 64, shuffle= True, num_workers= 2,pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
            myImageFloder(root = "./chromosomes_data/val", label = "./chromosomes_data/all_label.txt", 
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])), 
            batch_size= 20, shuffle= False, num_workers= 2,pin_memory=True)
    
    
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    """
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    """
    best_prec1 = 0
    for epoch in range(0, 80):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, resnet50, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, resnet50, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'Resnet50',
            'state_dict': resnet50.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = target.cuda(async=True)
        #input = input.to(cuda0)
        #target = target.to(cuda0)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(cuda0)
        target = target.to(cuda0)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()